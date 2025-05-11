# Лог публикации библиотеки

1. Собрали

```
uv build
```

2. Получили токен.

3. прописали в toml:

```
[[tool.uv.index]]
name = "testpypi"
url = "https://test.pypi.org/simple/"
publish-url = "https://test.pypi.org/legacy/"
explicit = true
```

4. Публикуем через токен

```
uv publish --index testpypi
```
5. Проверяем, что появился

```
uv run --with matrix_lib --no-project -- python -c "import matrix_lib"
```
