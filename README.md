# Petrolog AI

**Sun'iy intellekt asosida petrografik va petrologik tahlil platformasi**

Mikroskop ostidagi shlif va anshlif tasvirlarini, ICP-MS geokimyoviy ma'lumotlarini va 360° optik kuzatuvlarni yagona multimodal tizimda tahlil qilib, tayyor laboratoriya hisobotini shakllantiruvchi platforma.

🔗 **Jonli demo:** https://USERNAME.github.io/petrolog-ai/
📄 Loyiha: President AI Award 2026 · Yo'nalish: sanoat va biznesda AI

> `USERNAME` o'rniga o'z GitHub foydalanuvchi nomingizni yozing.

---

## Muammo

O'zbekistonda geologik-qidiruv ishlari hajmi o'sib bormoqda, lekin har bir shlif va anshlif hanuzgacha qo'lda tavsiflanadi:

- **Vaqt** — bitta namunaning to'liq petrografik tavsifi o'rtacha 3–5 ish kuni. H.M. Abdullayev nomidagi Geologiya va geofizika institutida 2024-yilda 1 800 ta, 2025-yilda 2 000 ta tahlil bajarildi.
- **Subyektivlik** — bir xil shlifni ikki mutaxassis turlicha tavsiflaydi; natijalar takrorlanuvchan emas.
- **Parchalanganlik** — shlif tasviri, anshlif, ICP-MS jadvali va ekspert xulosasi hech qachon bitta bazaga birlashtirilmaydi.

## Yechim

To'rtta manbadan kelgan ma'lumot bitta AI konveyerida qayta ishlanadi:

| Kirish ma'lumoti | AI nima qiladi | Chiqish |
|---|---|---|
| Shlif tasviri (o'tgan nur) | Mineral fazalarni segmentatsiyalash, tekstura tahlili | Mineral tarkib, % nisbat |
| Anshlif tasviri (qaytgan nur) | Ma'dan minerallarini tanish | Ruda mineralogiyasi |
| ICP-MS jadvali | Geokimyoviy tasnif, klark koeffitsiyentlari | Petrogenetik talqin |
| 360° pleoxroizm videosi | Optik xossalar o'zgarishini qayd etish | Identifikatsiyani tasdiqlash |

Foydalanuvchi to'rtta rejimdan birini tanlaydi: shlif tavsifi, anshlif tavsifi, ICP tahlili yoki **kompleks tahlil**. Kompleks rejimda barcha manbalar birga qayta ishlanib, yagona standartlashtirilgan hisobot shakllanadi.

---

## Ushbu repozitoriyda nima bor

Bu — platformaning **ishlaydigan demo prototipi** (`index.html`). Bitta faylda, serversiz, to'liq brauzer ichida ishlaydi.

**Demo nimani ko'rsatadi:**

1. Shlif va/yoki anshlif tasvirini yuklash
2. Fazalarni avtomatik ajratish — k-means klasterlash (RGB + lokal tekstura gradiyenti bo'yicha 4D xususiyat vektori, k-means++ initsializatsiyasi)
3. Har bir fazani optik xossalar reference jabvali orqali ehtimoliy mineral bilan moslashtirish va **ishonchlilik darajasi** berish
4. Faza xaritasini vizuallashtirish va foizli tarkibni hisoblash
5. ICP-MS ma'lumotlarini tahlil qilish — TAS bo'yicha dastlabki tasnif, A/CNK va A/NK indekslari, klark koeffitsiyentlari bo'yicha ma'danlashuv anomaliyalari
6. Avtomatik matnli petrografik tavsif generatsiyasi
7. Hisobotni yuklab olish

### Demo cheklovlari — ochiq aytamiz

Ushbu bosqichda **o'qitilgan neyron tarmoq ishlatilmaydi.** Segmentatsiya klasterlash algoritmiga, mineral nomlari esa reference jadvalga asoslangan.

Sababi oddiy: CNN modelini o'qitish uchun ekspert tomonidan tavsiflangan etalon dataset kerak, u esa loyihaning birinchi bosqichida yaratiladi. Demo **konveyerning uchdan-uchgacha ishlashini** isbotlaydi — aniqlik dataset bilan keladi.

Loyihaning asosiy ustunligi texnologiyada emas, **ma'lumotda**: CNN arxitekturasini har kim yozadi, O'zbekiston tog' jinslari bo'yicha ekspert tavsiflangan 5 000+ tahlil natijasiga esa mamlakatda boshqa hech kim ega emas.

---

## Yo'l xarita

| Bosqich | Muddat | Natija | KPI |
|---|---|---|---|
| Dataset yadrosi | 1–3-oy | 20+ ilmiy hisobotni raqamlashtirish, annotatsiya reglamenti | ≥1 500 annotatsiyalangan tasvir |
| Birinchi model | 4–6-oy | Segmentatsiya + klassifikatsiya modelini o'qitish | Asosiy jinshosil minerallar bo'yicha F1 ≥ 0,75 |
| Yopiq beta | 7–9-oy | Institut laboratoriyasida real ish oqimida sinov | ≥300 namuna tizim orqali o'tkazilgan |
| Pilot mijoz | 10–12-oy | Sanoat korxonasida pilot joriy etish | ≥1 shartnoma; vaqt tejash ≥60% |

Aniqlik bo'yicha yakuniy maqsadli ko'rsatkich pilot dataset natijalaridan keyin belgilanadi.

## Rejalashtirilgan texnologik stack

- **Model:** Python / PyTorch — segmentatsiya (U-Net / SegFormer sinfi), klassifikatsiya (CNN + confidence), transfer learning
- **Backend:** FastAPI
- **Frontend:** React
- **Ma'lumotlar:** PostgreSQL + obyekt saqlash
- **Yetkazish:** Docker (on-premise variant maxfiylik talabi yuqori mijozlar uchun)

**Human-in-the-loop majburiy:** ishonchlilik chegarasidan past natija «ekspert tekshiruvi zarur» belgisini oladi va hech qachon avtomatik tasdiqlanmaydi. Har bir ekspert tuzatishi keyingi o'qitish siklida datasetga qaytadi.

---

## Ishga tushirish

Hech qanday o'rnatish talab qilinmaydi:

```bash
git clone https://github.com/USERNAME/petrolog-ai.git
cd petrolog-ai
# index.html faylini brauzerda oching
```

Yoki jonli demo: https://USERNAME.github.io/petrolog-ai/

Sinash uchun istalgan shlif mikrofotosuratini yuklang va ICP-MS maydoniga namuna ma'lumot kiriting:

```
SiO2,62.4,%
Al2O3,15.8,%
Fe2O3,5.2,%
CaO,4.1,%
Na2O,3.6,%
K2O,2.9,%
MgO,2.4,%
Cu,340,ppm
Au,0.8,ppm
```

---

## Jamoa

Ilmiy yadro — H.M. Abdullayev nomidagi Geologiya va geofizika instituti: 1 nafar fan doktori va 5 nafar geologiya-mineralogiya fanlari falsafa doktori. NKMK, OKMK va «O'zbekgeologiya qidiruv» AJ bilan mineralogik-petrografik yo'nalishda hamkorlik tajribasi.

## Strategik asos

Loyiha O'zbekiston Respublikasi Prezidentining «Sun'iy intellekt texnologiyalarini 2030-yilgacha rivojlantirish strategiyasi» hamda PF-189-son Farmonining sun'iy intellektni iqtisodiyotning ustuvor tarmoqlariga joriy etish yo'nalishiga mos keladi.

---

## English summary

**Petrolog AI** is a multimodal AI platform for petrographic and petrological analysis of rocks. It combines thin-section images, polished-section images, ICP-MS geochemistry and 360° optical observation into a single analytical pipeline that produces a standardised laboratory report.

This repository contains a **working browser-based demo** (`index.html`, zero dependencies): upload a thin-section image, the app segments mineral phases via k-means clustering over RGB + local texture features, matches each phase against an optical-properties reference table with a confidence score, parses ICP-MS data for geochemical classification and ore anomalies, and generates a downloadable report.

The demo deliberately uses classical CV rather than a trained CNN — the expert-annotated reference dataset required for supervised training is the first deliverable of the project. The team's advantage is the data: 5 000+ expert-described rock analyses from 20+ completed research projects on Uzbek geology.

Submitted to **President AI Award 2026** (Uzbekistan), industry & business AI track.

---

## Litsenziya

Demo prototip kodi — MIT litsenziyasi (`LICENSE` fayliga qarang).

Loyihaning ma'lumotlar bazasi, ekspert annotatsiyalari va o'qitilgan model vaznlari MIT doirasiga **kirmaydi** va alohida intellektual mulk obyektlari hisoblanadi.
