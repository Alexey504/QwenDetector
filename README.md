# Qwen Detector — визуальный детектор на Qwen2.5-VL

Это оффлайн-веб-приложение, которое позволяет:
- загрузить изображение,
- написать текстовый запрос (например: «покажи, где машины»),
- получить исходное изображение с выделенными объектами и подписями.

## Пример:
<table>
  <tr>
    <td><img width="425" height="408" alt="Снимок экрана 2025-10-30 134315" src="https://github.com/user-attachments/assets/6a20c8f9-9467-4c87-b063-7ca5c21e1ef0"></td>
    <td><img width="428" height="402" alt="Снимок экрана 2025-10-30 133804" src="https://github.com/user-attachments/assets/a34bca86-b200-4ba2-864d-aa5f603df36f"></td>
  </tr>
</table>

## Быстрый старт

#### 1️) Клонировать репозиторий
```bash
git clone https://github.com/Alexey504/QwenDetector.git
cd QwenDetector
````

#### 2️) Скачать модель Qwen2.5-VL

Перед запуском необходимо **скачать модель вручную** с Hugging Face:

```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir ./models/Qwen2.5-VL-3B-Instruct
```

## Вариант 1. Запуск через Docker

#### 1️) Собрать образ

```bash
docker build -t qwen-detector .
```

#### 2️) Запустить контейнер

**GPU:**

```bash
docker run --gpus all -p 5050:5050 qwen-detector
```

**CPU:**

```bash
docker run -p 5050:5050 qwen-detector
```

#### 3️) Открыть приложение

Перейдите в браузере  [http://localhost:5050](http://localhost:5050)

---

## Вариант 2. Запуск без Docker

```bash
pip install -r requirements.txt
python app.py
```

---

## ⚙️ Структура проекта

```
qwen-detector/
├── app.py                     # Flask-приложение
├── detector/
│   ├── inference.py           # обработка изображений и текста
│   ├── model_loader.py        # загрузка локальной модели Qwen
│   └── __init__.py
├── templates/
│   └── index.html             # веб-интерфейс
├── models/
│   ├── .gitkeep               # папка для модели (пустая в репо)
│   └── Qwen2.5-VL-3B-Instruct/ (нужно добавить вручную)
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .dockerignore
└── README.md
```

## Используемые технологии

* [🧩 Transformers](https://huggingface.co/docs/transformers)
* [⚡ PyTorch](https://pytorch.org/)
* [🧠 Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
* [🧱 Flask](https://flask.palletsprojects.com/)
* [🖼️ Pillow + Supervision](https://github.com/roboflow/supervision)
