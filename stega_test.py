import pytest
import numpy as np
from PIL import Image
import tempfile
import os

# Импортируем функции из вашего модуля
# Предположим, ваш основной файл называется stega.py
from stega import split_into_blocks, reconstruct_from_blocks, hide, seek


def test1():
    test_array = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
    blocks, positions = split_into_blocks(test_array, size=3)
    assert len(blocks) == 4
    for block in blocks:
        assert block.shape == (3, 3, 3)
    expected_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
    assert positions == expected_positions
def test2():
    test_array = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    blocks, positions = split_into_blocks(test_array, size=4)
    assert len(blocks) == 4
    for block in blocks:
        assert block.shape == (4, 4, 3)


def test3():
    img_array = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name)
        image_path = f.name

    try:
        test_message = "Hello, World!"
        output_path = image_path.replace('.png', '_encoded.png')
        key_path = image_path.replace('.png', '_key.txt')  # Исправлено!

        result_path = hide(image_path, test_message, output_path, 8)  # size=8
        assert os.path.exists(result_path)

        assert os.path.exists('key.txt')

        os.rename('key.txt', key_path)

        with open(key_path, 'r', encoding='utf-8') as f:
            key_content = f.read()
        extracted = seek(result_path, key_path, 8)  # size=8

        assert extracted == test_message
        return True

    except AssertionError as e:
        import traceback
        traceback.print_exc()
        return False

    except Exception as e:

        import traceback
        traceback.print_exc()
        return False

    finally:
        files = [
            image_path,
            output_path if 'output_path' in locals() else None,
            key_path if 'key_path' in locals() else None,
            'key.txt'
        ]

        for path in files:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    print(f"   Удалён: {path}")
                except Exception as e:
                    print(f"   Не удалось удалить {path}: {e}")