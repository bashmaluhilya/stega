from PIL import Image
import numpy as np
import random
import sys

def split_into_blocks(img_array, size):
    """
    Разбивает множество пикселей на блоки размерами size x size
    :param img_array: множество пикселей
    :param size: размер блока
    :return: массив блоков и их координаты x, y внутри первоначального множества
    """
    height, width, channels = img_array.shape # передает параметры высоты, ширины и цвета
    blocks = []
    positions = []

    for y in range(0, height, size):
        for x in range(0, width, size):
            block = img_array[y:y+size, x:x+size, :]
            if block.shape[0] < size or block.shape[1] < size: #проверяет, находится ли блок у края
                padded_block = np.zeros((size, size, channels), dtype=img_array.dtype)  #создает блок нужного размера, запоненный 0
                padded_block[:block.shape[0], :block.shape[1], :] = block # вставляем в край наш неполный блок (остальное остается нулями)
                blocks.append(padded_block)
            else:
                blocks.append(block)
            positions.append((y, x)) #записываем координаты блока

    return blocks, positions

def reconstruct_from_blocks(blocks, positions, original_shape, size):
    """
    Собирает блоки в множество пикселей (функция, обратная split_into_blocks)
    :param blocks: массив блоков
    :param positions: массив координат блоков в первоначальном изображении
    :param original_shape: параметры первоначального изображения
    :param size: размер блока
    :return: возвращает множество пикселей изображения
    """
    height, width, channels = original_shape
    reconstructed = np.zeros((height, width, channels), dtype=blocks[0].dtype)

    for block, (y, x) in zip(blocks, positions): #одновременно проходимся по блокам и их координатам
        h = min(size, height - y)
        w = min(size, width - x)
        reconstructed[y:y + h, x:x + w, :] = block[:h, :w, :] #записываем блок в собираемое множество
    return reconstructed

def smoothness(block):
    """
    Функция оценивает гладкость блока
    :param block: обрабатываемый блок
    :return: возвращает параметр гладкости
    """
    h, w, channels = block.shape
    total_smoothness = 0.0
    for ch in range(channels): #считаем гладкость для каждого канала
        channel_data = block[:, :, ch]
        grad_x = np.abs(np.diff(channel_data, axis=1)) #модуль поэлементной разности элементов по оси x
        grad_y = np.abs(np.diff(channel_data, axis=0)) #модуль поэлементной разности элементов по оси y
        all_gradients = np.concatenate([grad_x.flatten(), grad_y.flatten()]) #превращает две матрицы в векторы и склевивает их в один больщой вектор

        if len(all_gradients) > 0:
            D_value = np.var(all_gradients) #вычисляет дисперсию
        else:
            D_value = 0.0
        total_smoothness += D_value

    return total_smoothness / 3.0

def get_random_positions(block_shape, n_bits, seed=None):
    """
    :param block_shape: параметры блока
    :param n_bits: количество бит возможных для кодировки
    :param seed:
    :return: возвращает список позиций в которых нужно кодировать биты
    """
    h, w, channels = block_shape #распаковывает блок

    if seed is not None: #seed - начальное значение для генератора псевдослучайных чисел
        random.seed(seed)
    all_positions = [(y, x, ch)
                     for y in range(h)
                     for x in range(w)
                     for ch in range(channels)]

    selected_positions = random.sample(all_positions, n_bits) #кодирует n бит в случайных позициях
    selected_positions.sort() # сортирует список

    return selected_positions

def hide(image, data, save_place, size, seed = None):
    """
    Кодирует текст в изображение
    :param image: кодируемое изображение
    :param data: текст, который необходимо закодировать
    :param save_place: место, куда необходимо сохранить новое изображение
    :param size: размер блоков
    :param seed:
    :return: возвращает изображение со встроинными в него битами текста и ключ
    """
    if seed is not None:
        random.seed(seed)
    key_parts = [] #массив в котором хранятся части ключа
    img = Image.open(image)
    img_array = np.array(img) #преобразуем изображение в массив Numpy
    hide_data = ''.join(format(ord(c), '08b') for c in data) #переводит текст в двоичный код
    all_bits = len(hide_data)
    blocks, positions = split_into_blocks(img_array, size)
    indices = list(range(len(blocks))) # список индексов всех блоков
    random.shuffle(indices) #перемешивает последовательность обхода блоков
    encoded_blocks = [block.copy() for block in blocks] #создаем копии блоков
    bit_index = 0

    for idx in indices:
        if bit_index >= all_bits:
            break

        block = blocks[idx]
        y, x = positions[idx]
        smooth_val = smoothness(block) #считаем гладкость

        # Определяем n по гладкости
        if 0.15 <= smooth_val < 0.3:
            n = 4
        elif 0.3 <= smooth_val < 2:
            n = 8
        elif 2 <= smooth_val < 7:
            n = 12
        elif smooth_val >= 7:
            n = 16
        else:
            n = 0

        if n == 0:
            continue

        remaining_bits = all_bits - bit_index # сколько бит осталось закодировать
        n = min(n, remaining_bits) #кодируем ровно столько бит, сколько нужно

        if bit_index == 0:
            print(f"Первые 8 бит: {hide_data[:8]}")
            print(f"Битовая строка для 'hello':")
            for i, char in enumerate(data):
                bits = format(ord(char), '08b')
                print(f"  '{char}' ({ord(char):3d}): {bits}")

        pos_seed = seed + idx if seed is not None else None #рандомизируем позицию блока
        random_positions = get_random_positions(block.shape, n, pos_seed)
        encoded_block = encoded_blocks[idx] #выбираем блок, который будем кодировать

        for i in range(n):
            if bit_index >= all_bits: #если все биты закодированы - выходим из цикла
                break
            pos_y, pos_x, pos_ch = random_positions[i] #извлекаем данные из рандомного блока
            bit = int(hide_data[bit_index]) #достаем кодируемый бит
            if bit_index < 8:
                print(f"  Кодирую бит {bit_index}: {bit} "
                      f"в блок {idx} позиция ({pos_y},{pos_x},{pos_ch})")

            old_value = encoded_block[pos_y, pos_x, pos_ch] #достаем старое значение
            new_value = (old_value & 0xFE) | bit #И с маской 11111110 + или с битом
            encoded_block[pos_y, pos_x, pos_ch] = new_value #вставляем биты в блок

            bit_index += 1

        positions_str = '|'.join([f'{y},{x},{ch}' for y, x, ch in random_positions]) # перебираем все позиции в блоке и превращаем в строку
        # кодируем в ключ индекс блока, количество кодируемых бит, координаты блока, координаты внутри блока
        key_parts.append(f"{idx}:{n}:{y}:{x}:{pos_seed}:{positions_str}")

    key = ';'.join(key_parts) #объединяем в один ключ
    reconstructed = reconstruct_from_blocks(encoded_blocks, positions, img_array.shape, size) #объединяем блоки в изображение
    encoded_img = Image.fromarray(reconstructed) #создаем изображение из массива
    encoded_img.save(save_place) #сохраняем
    with open('key.txt', 'w', encoding='utf-8') as f:
        f.write(key)
    return save_place

def seek(encoded_image, key, size):
    """
    Функция декодирует изображение
    :param encoded_image: декодируемое изображение
    :param key_file: ключ
    :param size: размер блоков
    :return:
    """
    img = Image.open(encoded_image)
    img_array = np.array(img)
    blocks, positions = split_into_blocks(img_array, size)
    key_parts = key.split(';') #разбивает ключ по разделителю
    extracted_bits = []

    print(f"Загружено блоков: {len(blocks)}")
    print(f"Частей в ключе: {len(key_parts)}")

    for part_num, part in enumerate(key_parts): #перебираем все части ключа с их номерами
        if not part.strip(): #проверяет не пустая ли строка
            continue

        segments = part.split(':') #перебираем часть ключа по двоеточиям
        if len(segments) < 6:
            print(f"Пропускаем некорректную часть {part_num}: {part}")
            continue
        #извлекаем инфу из ключа
        idx = int(segments[0])
        n = int(segments[1])
        positions_str = segments[5]

        if idx >= len(blocks):
            print(f"Ошибка: индекс блока {idx} >= {len(blocks)}")
            continue

        block = blocks[idx]

        if not positions_str:
            print(f"Пустая строка позиций в блоке {idx}")
            continue

        random_positions = []
        for pos_pair in positions_str.split('|'): #Разбиваем строку по разделителю
            if not pos_pair.strip(): #проверяет не пустая ли строка
                continue
            y, x, ch = map(int, pos_pair.split(',')) #разбиваем строки на составляющие
            random_positions.append((y, x, ch))

        if len(random_positions) != n:
            print(f"Предупреждение: блок {idx} - позиций {len(random_positions)}, ожидалось {n}")

        for i in range(min(n, len(random_positions))): #запускает цикл по минимальному значению
            pos_y, pos_x, pos_ch = random_positions[i]

            if pos_y >= size or pos_x >= size or pos_ch >= 3:
                print(f"Ошибка границ: {pos_y},{pos_x},{pos_ch} в блоке {size}x{size}x3")
                continue

            pixel_value = block[pos_y, pos_x, pos_ch] #получаем значение пикселя
            bit = pixel_value & 1 #извлекаем младший бит через битовую энд и маску 00000001
            extracted_bits.append(str(bit)) #добавляет бит в список

            if len(extracted_bits) <= 5:
                print(f"Блок {idx}, позиция {pos_y},{pos_x},{pos_ch}: "
                      f"значение={pixel_value}, LSB={bit}")

    bits_str = ''.join(extracted_bits) # собираем битовую строку
    print(f"\nВсего извлечено бит: {len(bits_str)}")

    if len(bits_str) < 40:
        print(f"Ошибка: недостаточно бит ({len(bits_str)}) для 5 символов")

    result_chars = []
    for i in range(0, len(bits_str), 8):
        byte_str = bits_str[i:i + 8]
        if len(byte_str) == 8:
            char_code = int(byte_str, 2) #преобразуем байт в десятичное число
            result_chars.append(chr(char_code)) #симлвол
            print(f"Байт {i // 8}: {byte_str} = {char_code} ('{chr(char_code)}')")
        else:
            print(f"Неполный байт: {byte_str}")

    result = ''.join(result_chars)
    print(f"\nДекодированное сообщение: '{result}'")
    return result

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Скрыть: python stega.py hide <image> <текст> <выходной_файл> <размер_блока> [seed]")
        print("Найти:  python stega.py seek <изображение> <ключ.txt> <размер_блока>")
        sys.exit()

    mode = sys.argv[1] #проверяем режим работы

    try:
        if mode == "hide":
            if len(sys.argv) < 6:
                raise ValueError("Нужно: hide <image> <текст> <output> <block_size> [seed]")

            img = sys.argv[2]
            data = sys.argv[3]
            output = sys.argv[4]
            size = int(sys.argv[5])

            # Обработка необязательного параметра seed
            seed = int(sys.argv[6]) if len(sys.argv) > 6 else None

            result_path = hide(img, data, output, size, seed)
            print(f"\n✓ Готово! Изображение сохранено: {result_path}")

        elif mode == "seek":
            if len(sys.argv) < 5:
                raise ValueError("Нужно: seek <image> <key.txt> <block_size>")

            img = sys.argv[2]
            key_file = sys.argv[3]
            size = int(sys.argv[4])

            # Читаем ключ из файла и передаём в seek
            with open(key_file, 'r', encoding='utf-8') as f:
                key = f.read()

            message = seek(img, key, size)
            print(f"\n Сообщение: '{message}'")

        else:
            print(f"Неизвестная команда: {mode}")
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Ошибка: файл не найден - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка значения: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

