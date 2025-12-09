from PIL import Image
import numpy as np
import random
image = "C:/Users/bashm/OneDrive/Desktop/AIP/stega/stega/500px-Lenna.png"
img = Image.open(image)
img_array = np.array(img)
data = 'hello, world'
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
    f = open('key.txt', 'w')
    f.write(key)
    return save_place, f


def seek(encoded_image, key, size):
    """

    :param encoded_image: декодируемое изображение
    :param key: ключ
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
        block_y = int(segments[2])
        block_x = int(segments[3])
        seed_str = segments[4]
        pos_seed = None if seed_str == 'None' else int(seed_str)
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
hide(image, data, "C:/Users/bashm/OneDrive/Desktop/AIP/stega/stega/encoded1_lena.png", 16)
#seek( "C:/Users/bashm/OneDrive/Desktop/AIP/stega/stega/encoded1_lena.png", "482:16:240:32:None:0,3,2|0,6,0|1,9,1|3,0,1|3,12,1|4,2,2|6,2,2|7,10,1|8,10,2|8,15,2|9,1,2|9,14,1|11,10,2|12,6,0|13,15,1|15,0,0;822:16:400:352:None:1,8,1|1,9,1|3,6,2|3,15,1|4,7,0|5,13,0|7,2,1|9,3,2|11,1,0|11,15,0|12,0,1|13,1,2|13,5,0|14,7,0|14,13,0|15,9,0;259:16:128:48:None:0,9,2|1,1,2|1,3,2|1,9,1|4,3,0|4,4,2|5,5,1|5,6,1|5,13,2|7,2,1|9,11,0|10,14,2|11,15,2|12,0,2|14,13,1|14,14,1;591:16:288:240:None:0,9,0|1,6,1|2,2,1|4,5,2|5,1,0|5,6,2|6,14,1|9,4,2|10,7,1|10,13,2|11,2,2|11,10,2|12,6,2|12,8,2|12,12,2|15,1,0;567:16:272:368:None:0,7,1|4,5,2|4,6,1|5,9,0|6,10,0|6,10,1|6,11,0|7,4,1|7,15,1|9,11,1|10,13,1|10,15,1|12,8,0|13,2,1|14,15,1|15,11,2;982:16:480:352:None:0,6,2|2,11,0|2,15,0|2,15,2|3,2,2|3,3,2|3,11,0|5,8,1|9,13,0|10,12,1|12,4,0|12,8,1|13,4,2|14,1,2|14,7,2|15,7,1", 16)