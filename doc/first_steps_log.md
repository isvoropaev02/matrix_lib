# Лог создания библиотеки `matrix_lib` (пока будет здесь)

Нужен для отладки и воспроизведения процесса создания библиотеки с нуля.

## Проделанные шаги (Windows)

**1. Установка `uv`:**

```
pip install uv
```

**2. Создание шаблона библиоткеи:**

```
uv init --lib matrix_lib
```

Создается директория со следующим содержанием:

```
matrix_lib
|   .git
|   .gitignore
|   .python-version
|   pyproject.toml
|   README.md
|
\---src
    \---matrix_lib
            py.typed
            __init__.py
```

**3. Добавим пакет `numpy`:**

```
uv add numpy
```

Автоматически было создано витруальное окружение `.venv`.

**4. пробуем запустить питон с помощью `uv` через виртуальное окружение:**

Создали тестовый файлик `test_run.py`:

```
print("Hello from local directory!")
from matrix_lib import hello
print(hello())
```
И запустили:
```
uv run python test_run.py
# output
# Hello from local directory!
# Hello from matrix-lib!
```

**5. Активация виртуального окружения (PowerShell)**

```
.\.venv\Scripts\Activate.ps1
```

**6. Переписываем в файле `uv.lock` источник c PyPi на тестовый:**

```
source = { registry = "https://test.pypi.org/simple" }
```
**7. Планы на гит:**

- Будет ветка `prod`, которую надо защитить от изменений. Изменения можно будет вносить только с ветки `develop`.

- В ветке `develop` надо организовать CI/CD с помощью Github Actions (2000 минут бесплатных есть).

- В ветку `develop` можно будет заливать код только через pull request.

- В ветку `prod` будет заливаться только стабильный `develop`. Там уже не будет CI/CD.

- Публиковать библиотеку в test.pypi только с ветки `prod`.

**8. Настройка pre-commit:**

```
# Установка pre-commit через uv
uv pip install pre-commit
```

Создаем файл конфигурации `.pre-commit-config.yaml`.

В активированном окружении запускаем:

```
pre-commit install
pre-commit run --all-files
```

**8. Создаем базовый класс `Matrix` на новой векте векте:**

Потом надо пробовать бедать pull request. Далее будет PR в истории на гитхабе. Заодно проверим работу `squash commits`
