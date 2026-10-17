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
curl -H "X-API-Key: $KEY" localhost:8000/scans/<id>            # статус, результат, тексты RU/UZ
curl -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"decision": "corrected", "paper_speed_mm_s": 50}' localhost:8000/scans/<id>/confirm
curl -H "X-API-Key: $KEY" -X POST localhost:8000/scans/<id>/analysis   # заключение CardioLens
```

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
