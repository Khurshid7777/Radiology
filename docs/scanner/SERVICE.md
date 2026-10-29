# Сервис сканера и Telegram-бот (M4)

## Запуск API

```bash
pip install -e ".[scanner]"
export ECGSCAN_API_KEYS="bot:<ключ бота>,dr_ivanova:<ключ врача>"     # без ключей API отвечает 503
export ECGSCAN_STORAGE_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
uvicorn cardiolens.api.app:app        # CardioLens + /scans
# или только сканер:  uvicorn ecgscan_service.app:app
```

Ключ хранилища нужно сохранить: без него зашифрованные оригиналы не прочитать. Если ключ не задан,
оригиналы не хранятся вовсе (см. SD-040). Настройки — `configs/scanner/service.yaml`.

## Пример

```bash
curl -H "X-API-Key: $KEY" -F file=@ecg.jpg -F client=api localhost:8000/scans
# {"scan_id": "…", "status": "queued"}
# Лист на подложке, снятый по частям (SD-066): первая часть в file, остальные в parts (до 4 снимков)
curl -H "X-API-Key: $KEY" -F file=@left.jpg -F parts=@right.jpg localhost:8000/scans
curl -H "X-API-Key: $KEY" localhost:8000/scans/<id>            # статус, результат, тексты RU/UZ
curl -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"decision": "corrected", "paper_speed_mm_s": 50}' localhost:8000/scans/<id>/confirm
curl -H "X-API-Key: $KEY" -X POST localhost:8000/scans/<id>/analysis   # заключение CardioLens
```

### Возможности сервера (P2, SD-097)

`GET /scans/config/features` → `{"features": {"consult": false, "archive": false, "qr": false,
"signature": false}}`. Сейчас все выключены — см. [EXTENSIONS.md](EXTENSIONS.md).

### Выгрузка сигнала (S4, SD-094)

```bash
curl -H "X-API-Key: $KEY" -OJ "localhost:8000/scans/<id>/export?format=wfdb"   # ecg-<id8>.zip: ecg.hea + ecg.dat
curl -H "X-API-Key: $KEY" -OJ "localhost:8000/scans/<id>/export?format=csv"    # time_s и по столбцу на канал
curl -H "X-API-Key: $KEY" -OJ "localhost:8000/scans/<id>/export"               # JSON контракта ScanResult
```

- **Ось одна:** все каналы на общей оси времени с частотой `sampling_rate_hz`, от t = 0.
- **Порядок каналов:** 12 отведений в стандартном порядке, затем полосы ритма (`II_rhythm`).
- **Пропуски:** отведение есть только в своём окне листа (на 3x4 — 2,5 с). Вне окна значение
  пустое (CSV) или недопустимое (WFDB), а не ноль.
- **Точность WFDB:** формат 32 при 100 000 единиц/мВ, то есть шаг 0,01 мкВ, точность самого
  контракта.
- **Без потерь:** CSV и WFDB читаются обратно бит в бит
  (`ecgscan.export.read_csv` / `read_wfdb_zip`).
- **Комментарии WFDB:** статус, скорость и усиление с источником, раскладка и предупреждения.
  Персональных данных нет.
- **Ошибки:**
  - чужой снимок — 404;
  - неизвестный формат — 422;
  - снимок ещё без трасс — 409.
- **Журнал:** каждая выгрузка пишется в журнал действий (`exported`, формат).

## Бот

```bash
pip install -e ".[scanner,bot]"
export ECGSCAN_BOT_TOKEN=<токен от @BotFather>
export ECGSCAN_BOT_API_KEY=<ключ бота из ECGSCAN_API_KEYS>
export ECGSCAN_BOT_ALLOWED=<telegram user_id через запятую>
python -m ecgscan_bot
```

Адрес API и таймауты — `configs/scanner/bot.yaml`. ЭКГ отправляется **файлом** (скрепка → «Файл»).
Язык ответа: узбекский для клиентов Telegram на узбекском, иначе русский.

## Тексты

Все тексты статусов, кодов отказа и предупреждений, а также бота лежат в
`configs/scanner/i18n/{ru,uz}.json`. Узбекские тексты (латиница) нужно проверить с носителем языка.
