# Развёртывание CardioLens на сервер (D153)

Что поднимается: API и веб-страница загрузки CardioLens за Caddy (HTTPS автоматически).
Бот Telegram — позже.

## 1. Один раз: что сделать владельцу

1. Сервер: Ubuntu 22.04/24.04 или Debian 12, ≥ 2 ГБ ОЗУ (лучше 4), ≥ 20 ГБ диска. В DNS домена A-запись →
   IP сервера.
2. Сообщить главной сессии домен и IP — она впишет их в `deploy/target.env` (не секрет: DNS и так
   показывает IP).
3. В консоли сервера (под `root`) одной командой создать ключ выкладки, разрешить его и вывести на экран:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/cardiolens_deploy -N "" -q && cat ~/.ssh/cardiolens_deploy.pub >> ~/.ssh/authorized_keys && base64 -w0 ~/.ssh/cardiolens_deploy; echo
   ```
   Команда выводит закрытый ключ одной строкой (base64) — её проще скопировать целиком. Подходит и
   обычный вид (`cat ~/.ssh/cardiolens_deploy`, от `-----BEGIN` до `-----END`). Не подходит `.pub`.
4. Открыть https://github.com/Khurshid7777/Radiology/settings/secrets/actions/new — Name: `SERVER_SSH_KEY`,
   Secret: строка, которую вывела команда → Add secret.
5. Необязательно: так же секрет `ANTHROPIC_API_KEY` (ключ с console.anthropic.com; подписка claude.ai его
   не даёт). Без него нарратив — офлайн-шаблон.

## 2. Что делается само

Docker ставится при первой выкладке. Ключ клиники для `/scans` и ключ шифрования снимков создаются на
сервере один раз (`deploy/server_env.sh`) и через GitHub не проходят. Посмотреть их:
`cat /opt/cardiolens/deploy/.env`. Файл не удалять: без `ECGSCAN_STORAGE_KEY` старые снимки не
расшифровать.

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
