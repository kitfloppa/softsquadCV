## Installation Python packages

```bash
python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Task № 1

Все предобработанные данные сохраняются в папку «rgb_data»

## Task № 2

Данные хранятся в папке «input_dir_cars» (description.csv, data, image_counter.txt), туда же сохранятся файл «output_cars.csv» с результатами.

## Task № 3

Реализована функция «calc_metric» для нахождения доминирующего цвета в выделенной области.
