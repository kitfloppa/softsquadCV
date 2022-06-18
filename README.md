## Installation Python packages

```bash
python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Task № 1

Все предобработанные данные сохраняются в папку «rgb_data»

## Task № 2

Результат сохранятся в файл «output_cars.csv».

## Task № 3

Реализована функция «calc_metric» для нахождения доминирующего цвета в выделенной области.

## Task № 4

Реализована функция «find_color» для опредления цвета машины, результаты записываются в «output_color.csv»

## Task № 5

Обучена модель «yolov5s» для определения включенных набаритных огней на авто (веса «lights.pt»)

```bash
    git clone "https://github.com/ultralytics/yolov5.git"
```

Реализована функция «check_stop_signals» для опредления цвета машины, результаты записываются в «output_lights.csv»