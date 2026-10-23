# Развёртывание CardioLens на сервер (D153)

Что поднимается: API и веб-страница загрузки CardioLens за Caddy (HTTPS автоматически).
Бот Telegram — позже.

## 1. Один раз: сервер и домен

1. Сервер: Ubuntu 22.04/24.04 или Debian 12, ≥ 2 ГБ ОЗУ (лучше 4), ≥ 20 ГБ диска.
2. В DNS домена A-запись `@` (или поддомен, например `ecg`) → IP сервера. Порты 80 и 443 открыты.
3. Ключ SSH для выкладки (на своём компьютере):
   ```bash
   ssh-keygen -t ed25519 -f cardiolens_deploy -N ""
   ssh-copy-id -i cardiolens_deploy.pub root@IP_СЕРВЕРА
   ```
   Пользователь — `root` или с `sudo` без пароля. Docker скрипт `deploy/server_setup.sh` поставит сам.

## 2. Один раз: секреты в GitHub

Репозиторий → Settings → Environments → **New environment** `production`, в нём **Add secret**:

| Секрет | Что вписать |
|---|---|
| `SERVER_HOST` | IP сервера |
| `SERVER_USER` | `root` (или пользователь с sudo) |
| `SERVER_SSH_KEY` | содержимое файла `cardiolens_deploy` (закрытый ключ целиком) |
| `DOMAIN` | домен, например `ecg.example.uz` |
| `ECGSCAN_API_KEYS` | ключи клиник для `/scans`: `clinic1:длинный_случайный_ключ` (через запятую) |
| `ECGSCAN_STORAGE_KEY` | ключ шифрования снимков: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `ANTHROPIC_API_KEY` | необязательно; ключ из console.anthropic.com. Без него нарратив — офлайн-шаблон |

`ECGSCAN_STORAGE_KEY` после первой выкладки не менять: старые снимки перестанут расшифровываться.

## 3. Выкладка и обновление

Перед каждой выкладкой — подтверждение владельца. Затем:

GitHub → Actions → **Deploy** → Run workflow → `ref` = `main` (или нужная ветка/коммит).

Workflow проверяет, что CI на коммите зелёный, копирует код на сервер в `/opt/cardiolens`, собирает
образ и перезапускает контейнеры, затем ждёт ответа `https://DOMAIN/health`. Данные (`/data` в томе
`appdata`) при обновлении сохраняются.

## 4. На сервере вручную (если нужно)

```bash
cd /opt/cardiolens
docker compose -f deploy/docker-compose.yml --env-file deploy/.env ps
docker compose -f deploy/docker-compose.yml --env-file deploy/.env logs -f cardiolens
docker compose -f deploy/docker-compose.yml --env-file deploy/.env restart
```

Откат: запустить Deploy с `ref` = предыдущий коммит.
