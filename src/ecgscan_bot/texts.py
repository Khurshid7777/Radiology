"""The D208 bot texts (RU / UZ). The older texts live in ``ecgscan.i18n``."""

from __future__ import annotations

_T = {
    "need_code": (("Доступ к боту — по коду приглашения. Отправьте: /code ВАШКОД\n"
                  "Код выдаёт администратор CardioVision. Ваш Telegram id: {user_id}"),
                  ("Botdan foydalanish — taklif kodi bilan. Yuboring: /code KODINGIZ\n"
                  "Kodni CardioVision administratori beradi. Telegram id: {user_id}")),
    "code_ok": ("Код принят — можно присылать ЭКГ файлом (фото, PDF, файл аппарата).",
                "Kod qabul qilindi — EKGni fayl sifatida yuborishingiz mumkin."),
    "code_bad": ("Код не подходит или отозван. Проверьте код у администратора.",
                 "Kod mos emas yoki bekor qilingan. Administrator bilan tekshiring."),
    "group": ("Бот работает только в личном чате: в группе ответ увидят все участники.",
              "Bot faqat shaxsiy chatda ishlaydi."),
    "admin_only": ("Команда только для администратора.", "Faqat administrator uchun."),
    "invite": ("Новый код: {code} (метка: {label})\nСсылка для входа: {link}\nОтозвать: /revoke {code}",
               "Yangi kod: {code} ({label})\nKirish havolasi: {link}\nBekor qilish: /revoke {code}"),
    "codes_none": ("Кодов пока нет. Новый: /invite метка", "Kodlar yo'q. Yangi: /invite belgi"),
    "revoked": ("Код {code} отозван: вошедшие по нему больше не могут присылать ЭКГ.",
                "{code} kodi bekor qilindi."),
    "revoke_unknown": ("Такого кода нет.", "Bunday kod yo'q."),
    "actions": ("Что ещё сделать с этим анализом:", "Bu tahlil bilan yana nima qilish mumkin:"),
    "btn_symptoms": ("Указать симптомы", "Belgilarni ko'rsatish"),
    "btn_share": ("Ссылка для коллеги", "Hamkasb uchun havola"),
    "btn_lang": ("O'zbekcha", "Русский"),
    "btn_recalc_sym": ("Пересчитать с симптомами", "Belgilar bilan qayta hisoblash"),
    "symptoms_pick": ("Отметьте симптомы и когда они начались, затем «Пересчитать с симптомами».",
                      "Belgilarni va boshlanish vaqtini belgilang, so'ng «qayta hisoblash»."),
    "share": ("Ссылка для коллеги (PDF без фото, действует 7 дней):\n{url}",
              "Hamkasb uchun havola (rasmsiz PDF, 7 kun amal qiladi):\n{url}"),
    "share_fail": ("Ссылку сделать не удалось — анализ больше не хранится или сервер недоступен.",
                   "Havola yaratilmadi — tahlil saqlanmagan yoki server mavjud emas."),
    "ask_patient": (("Укажите пол и возраст пациента, например: «54 м», «ж 7», «3 мес м».\n"
                    "Если не знаете — нажмите «Пропустить» (тогда нормы взрослого)."),
                    ("Bemorning jinsi va yoshini yozing, masalan: «54 e», «a 7», «3 oy e».\n"
                    "Bilmasangiz — «O'tkazib yuborish» (unda kattalar me'yori).")),
    "ask_patient_again": ("Не понял ответ. Напишите, например, «54 м» или «ж 7», или нажмите «Пропустить».",
                          "Javobni tushunmadim. Masalan «54 e» yoki «a 7» yozing, yoki «O'tkazib yuborish»."),
    "ask_note": (("Есть ли диагноз или описание (заключение врача, аппарата, клиника)? Напишите текст "
                 "или нажмите «Нет». Не пишите ФИО пациента."),
                 ("Tashxis yoki tavsif bormi (shifokor, apparat xulosasi, klinika)? Matn yozing yoki «Yo'q» "
                 "tugmasini bosing. Bemor F.I.Sh.ni yozmang.")),
    "compressed": (("⚠️ Это сжатая копия: Telegram уменьшает фото, и мелкие зубцы теряются — заключение будет "
                    "ориентировочным. Для точного анализа пришлите оригинал файлом:\n"
                    "• Android: 📎 → «Файл» → «Галерея» → выберите снимок\n"
                    "• iPhone: 📎 → «Файл» → «Выбрать фото или видео» → выберите снимок\n"
                    "Сейчас разберу и эту копию."),
                   ("⚠️ Bu siqilgan nusxa: Telegram suratni kichraytiradi, mayda tishchalar yo'qoladi — xulosa "
                    "taxminiy bo'ladi. Aniq tahlil uchun asl nusxani fayl sifatida yuboring:\n"
                    "• Android: 📎 → «Fayl» → «Galereya» → suratni tanlang\n"
                    "• iPhone: 📎 → «Fayl» → «Rasm yoki video tanlash» → suratni tanlang\n"
                    "Hozir bu nusxani ham tahlil qilaman.")),
    "btn_skip": ("Пропустить", "O'tkazib yuborish"),
    "btn_no": ("Нет", "Yo'q"),
    "analysing": ("Анализирую…", "Tahlil qilinmoqda…"),
    "pending_expired": ("Вопросы устарели — пришлите ЭКГ ещё раз.", "Savollar eskirdi — EKGni qayta yuboring."),
    "expired": ("Этот ответ устарел — пришлите ЭКГ ещё раз.", "Bu javob eskirgan — EKGni qayta yuboring."),
}

SYMPTOM_BTN = {"chest_pain": ("Боль в груди", "Ko'krakda og'riq"), "dyspnea": ("Одышка", "Hansirash"),
               "syncope": ("Обморок", "Hushdan ketish"), "palpitations": ("Сердцебиение", "Yurak urishi")}
ONSET_BTN = {"lt12h": ("< 12 ч", "< 12 soat"), "12to48h": ("12–48 ч", "12–48 soat"),
             "gt48h": ("> 48 ч", "> 48 soat"), "unknown": ("не знаю", "noma'lum")}


def tr(key: str, lang: str, **kw) -> str:
    ru, uz = _T[key]
    return (uz if lang == "uz" else ru).format(**kw)
