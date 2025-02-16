# FactChecker Backend Dokumentáció

## Bevezetés

Ez a dokumentáció a FactChecker backend rendszerét mutatja be, amely egy Django-alapú REST API alkalmazás.
A backend szolgáltatja ki a Vue.js frontend számára szükséges adatokat és funkcionalitást, illetve kezeli az AI-alapú tényellenőrző rendszer működését kezdve az adatok begyűjtésétől egészen azok struktúrált visszaadásáig.

## Használt Stack-ek

### Alapvető Technológiák

- **Python**: v3.12.1 A backend fő programozási nyelve
- **Django**: v5.1.2 - Web framework
- **Django REST Framework**: v3.15.2 - REST API implementáció
- **Django Environ**: v0.11.2 - Környezeti változók kezelése
- **Django CORS Headers**: v4.5.0 - CORS kezelés

### Adatbázisok és Tárolás

- **PostgreSQL**: v16.3 - Elsődleges adatbázis
- **Neo4j**: v5.26.1 Community Edition - Knowledge graph adatbázis
- **pgvector**: v0.3.6 - Vektor tárolás és keresés PostgreSQL-ben

### AI és NLP Komponensek

- **LangChain**: v0.3.7 - AI modellek integrációja
- **OpenAI**: v1.60.0 - LLM szolgáltatások
- **Spacy**: v3.8.4 - Természetes nyelvfeldolgozás
- **HuSpacy**: v0.5.1 - Magyar nyelvű NLP modell
- **Sentence Transformers**: v3.4.1 - Szöveg beágyazások

### Infrastruktúra

- **Szerver**: Fedora (Szaniszló) szerver
- **Gunicorn**: v23.0.0 - WSGI HTTP Szerver
- **Uvicorn**: v0.32.0 - ASGI szerver implementáció

### Egyéb Függőségek

- **NumPy**: v1.26.4 - Numerikus műveletek
- **Pandas**: v2.2.3 - Adatfeldolgozás
- **PyJWT**: v2.10.1 - JWT token kezelés
- **Requests**: v2.32.3 - HTTP kliens
- **Python-dotenv**: v1.0.1 - Környezeti változók kezelése

A rendszer a "Szaniszló" becenevű Fedora szerveren fut, ahol a PostgreSQL és Neo4j szolgáltatások is üzemelnek.

## Telepítési útmutató

### Telepítés lépései

1. Projekt klónozása:

```bash
git clone https://github.com/MrZArmin/fact-checker.git
cd fact-checker
```

2. Venv létrehozása
```bash
python -m venv venv
source venv/bin/activate
```

3. Függőségek telepítése
```bash
pip install -r requirements.txt
```

4. Környezeti változók beállítása
```bash
cp .env.example .env
```

Ezt követően feltölteni a titkosított fájlban található értékekkel.

5. PGSQL letöltése és beállítása (Opcionális)
Javasolt a már meglévő szerveren levő adatbázist használni, így nem kell a lokális adabázis felállításával bajlódni.
Így persze kevésbé tesztelhető a folyamat az elejétől.
  - Az adott operációs rendszerre letölthető a PostgreSQL [innen](https://www.postgresql.org/download/)
  A projektben használt verzió: v16.3
  - Felhasználó és adatbázis létrehozása
  - Csatlakozási információ felvitele a `.env` fájlba
  - [PgVector letöltése és beállítása](https://github.com/pgvector/pgvector)
  - Migrációk futtatása
    - `python manage.py makemigrations`
    - `python manage.py migrate`
  
Az adatok vizualizációjához és az adatbázis grafikus kezeléséhez a [DBeaver](https://dbeaver.io/download/) alkalmazás ajánlott

6. Neo4j letöltése és beállítása (Opcionális)
Még kevésbé javasolt, a rendszer felállítása is macerás, feltölteni adatokkal pedig idő- és pénzigényes.

7. Szerver futtatása
```bash
python manage.py runserver
```

Amennyiben tesztelni szeretnénk a frondendet a lokális backenddel, ne felejtsük el átállítani ott az erre vonatkozó környezeti változót

### A rendszer fő funkciói:

#### Tényként szolgáló adatok el- és előkészítése

- Cikkek scrapelése az MTI weboldalról Selenium használatával
  - Először linkek scrapelése majd eltárolása adatbázisban
  - Linkek alapján több szimultán webdriver scrapeli a cikkeket
- Cikkek megtisztítása, a feleslegesek eldobása
- Adatok eltárolása PostgresSQL adatbázisban

#### Adatok feldolgozása RAG-hoz

- Adatok felchunkolása szemantikailag
- PgVector setup adatbázisban
- Adatok embeddelése különböző modellek használatával

#### Knowledge graph-ek

- Neo4j szerver
- Entitások kinyerése a cikkekből, azok eltárolása
- Query alapján relációs adatok vissszakérése az adatbázisból

#### Rendszer teljesítményének mérése

- Eldöntendő kérdések generálása LLM-mel a cikkek alapján
- Ezen az kérdésbankon futtatni a tesztelendő rendszert

#### RESTful API

- Felhasználói autentikáció és jogosultságkezelés
- Chat session-ök és üzenetek kezelése
- AI modellek integrációja és kommunikáció
- RESTful API végpontok biztosítása a frontend számára
