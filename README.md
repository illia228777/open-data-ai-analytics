# Open Data AI Analytics

## Назва проєкту
Аналіз відкритих державних даних про реєстрацію транспортних засобів в Україні (2022 рік)

## Мета

Створення модульного аналітичного проєкту з використанням Python та системи контролю версій Git.

Мета роботи - реалізувати повний цикл обробки відкритих даних:
від завантаження та перевірки якості до дослідження та візуалізації результатів.

## Джерело даних

Портал відкритих даних України:  
https://data.gov.ua/dataset/0ffd8b75-0628-48cc-952a-9302f9799ec0/resource/b1bcb4a9-8e60-4a1c-91c0-00faae008816/download/reestrTZ2022.zip

Набір даних: Відомості щодо транспортних засобів

Період даних:  
01.01.2022 – 31.12.2022

## Дослідницькі питання

1. **Чи існують статистично значущі регіональні відмінності у структурі автопарку України у 2022 році?**
2. **Чи формуються природні структурні кластери транспортних засобів за технічними характеристиками (об’єм двигуна, маса, тип кузова, вид пального)?**
3. **Чи відрізняється структура автопарку фізичних та юридичних осіб?**

## Загальна структура проєкту

Проєкт складається з таких модулів:

- `data_load` — завантаження та зчитування даних
- `data_quality_analysis` — перевірка якості та підготовка даних
- `data_research` — проведення аналітичних розрахунків
- `visualization` — побудова графіків і візуалізацій

---

## Запуск модуля завантаження даних

Для завантаження та первинного зчитування даних реалізовано CLI-команду `data-load`.

Приклад запуску:

```bash
uv run python -m oda_analytics data-load \
  --csv data/raw/<file_name>.csv \
  --nrows 100000
```

## Запуск модуля перевірки якості даних

Для виконання базової перевірки якості даних реалізовано CLI-команду `data-quality`.

Модуль здійснює аналіз пропущених значень, пошук дублікатів та базову перевірку числових показників (діапазони значень, мінімум/максимум).

Приклад запуску:

```bash
uv run python -m analytics data-quality \
  --input data/raw/<file_name>.parquet
```

## Запуск модуля аналізу даних

Для аналізу даних та проведення кластеризації реалізовано CLI-команду `data-research`.

Приклад запуску:

```bash
uv run python -m analytics data-research \
  --input data/raw/<file_name>.parquet
```

## Запуск модуля візуалізації даних

Для візуалізації даних реалізовано CLI-команду `data-visualize`.

Приклад запуску:

```bash
uv run python -m analytics data-visualize \
  --input data/raw/<file_name>.parquet
```


---

## Running with Docker

The pipeline runs end-to-end in containers: Postgres as the data store, each module as its own image, orchestrated by Compose.

### Prerequisites

- Docker + Docker Compose
- A copy of the source CSV at `./data/raw/vehicles_2022.csv`

### Setup

```bash
cp .env.example .env
# edit .env if you want a non-default password
```

### Run the full pipeline

```bash
docker compose up --build
```

Order of execution (compose handles this automatically):

1. `db` (Postgres 16) starts and becomes healthy.
2. `data_load` streams the CSV into Postgres via COPY.
3. `data_quality`, `data_research`, `visualization` run in parallel, reading from the DB.
4. `web` serves `pages/index.html` and the figures on http://localhost:8000.

### Stopping

```bash
docker compose down          # keep DB volume
docker compose down -v       # also drop DB and figures
```

### Services

| Service        | Dockerfile                 | Role                                       |
|----------------|----------------------------|--------------------------------------------|
| `db`           | `postgres:16` image        | Data store                                 |
| `data_load`    | `Dockerfile.data_load`     | CSV → Postgres via Polars + psycopg COPY   |
| `data_quality` | `Dockerfile.data_quality`  | Quality report over `vehicles` table       |
| `data_research`| `Dockerfile.data_research` | Stats + clustering on a random sample      |
| `visualization`| `Dockerfile.visualization` | Writes PNGs to the `figures` volume        |
| `web`          | `Dockerfile.web`           | Static HTTP server for figures + index.html|