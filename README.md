# Инструкция для пользователя

**Инструкция по установке:**

Используя `pip` поставить библиотеку следующей командой:

```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ matrix_lib
```

**Примеры для ознакомления с библиотекой:**

1. Hello World Example: [jupyter-notebook](examples\example_1_hello_world.ipynb)
2. Full Matrix Example: [jupyter-notebook](examples\example_2_full_matrix.ipynb)

# Инструкция для разработчика

## Установка проекта

**1. Клонируем векту `develop` из репозитория и переходим в директорию:**

```
git clone -b develop https://github.com/isvoropaev02/matrix_lib.git
cd matrix_lib
```

**2. Установка `uv` и подтягиваем нужные зависимости:**

По [официальной инструкции](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2).

И делаем:

```
uv sync --dev
```

**3. Пробуем запустить самоый простой ut:**

В таком режиме можно отлаживать код на каком-то одном юнит-тесте.

```
uv run python ut/test_import.py
```

**4. Пробуем запустить юнит тесты с помощью `pytest`:**

```
uv run pytest
```

**5. Пробуем запустить `pre-commit`:**

```
uv run pre-commit run --all-files
```

## Работа с git:

1. Можно открывать pull request в векту `develop`. Смерджиться можно после получения 1 апрува.

2. Желательно удалить векту, которая была смерджена с `develop` на сервере.

3. Правила названия пул реквестов:

```
[TYPE] DESCRIPTION
```
В скобочках указыватся тип изменений. Он может быть следующим:

- [FEATURE] - если был реализован новый функционал в `src\matrix_lib` (и даже если вместе с ним написан юнит тест).
- [TEST] - если функционал не добавлялся совсем, а был добавлен только тест.
- [BUILD] - если изменения касаются сборки, зависимостей, конфигов и инструкций проекта.
- [FIX] - ну не дай бог кто-то что-то уронил

4. Желательно при мердже делать 'squash commits' и удалять использованную ветку с сервера
