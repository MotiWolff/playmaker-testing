-- =========================
-- בסיס: קבוצות
-- =========================
CREATE TABLE team (
  team_id        SERIAL PRIMARY KEY,
  name           TEXT NOT NULL UNIQUE
);

-- =========================
-- RAW: שורות מקור מה-CSV (football-data.co.uk)
-- כולל כל העמודות שהקוד משתמש בהן + odds/סטטיסטיקות
-- =========================
CREATE TABLE match_raw (
  raw_id         BIGSERIAL PRIMARY KEY,
  src_file       TEXT NOT NULL,                         -- מזהה מקור/URL הקובץ
  row_hash       TEXT NOT NULL UNIQUE,                  -- SHA256 למניעת כפילויות

  -- עמודות CSV עיקריות שהקוד משתמש בהן:
  "Date"         DATE,                                  -- תאריך המשחק (כמו בקובץ)
  "HomeTeam"     TEXT,
  "AwayTeam"     TEXT,
  "FTHG"         INT,                                   -- Full Time Home Goals
  "FTAG"         INT,                                   -- Full Time Away Goals
  "FTR"          CHAR(1),                               -- 'H' / 'D' / 'A'
  "HS"           INT,                                   -- Home Shots
  "AS"           INT,                                   -- Away Shots
  "HC"           INT,                                   -- Home Corners
  "AC"           INT,                                   -- Away Corners
  "HY"           INT,                                   -- Home Yellows
  "AY"           INT,                                   -- Away Yellows
  "HR"           INT,                                   -- Home Reds
  "AR"           INT,                                   -- Away Reds

  -- Bet365 odds (מופיע בקוד):
  "B365H"        NUMERIC(8,3),                          -- Home win odds
  "B365D"        NUMERIC(8,3),                          -- Draw odds
  "B365A"        NUMERIC(8,3),                          -- Away win odds
  "B365BTSY"     NUMERIC(8,3),                          -- Both teams to score - Yes (אם קיים)
  "B365BTSN"     NUMERIC(8,3),                          -- Both teams to score - No (אם קיים)

  extras         JSONB DEFAULT '{}'::jsonb,             -- כל טור נוסף מהקובץ (לשמירת שלמות)
  ingested_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON match_raw ("Date");
CREATE INDEX ON match_raw ("HomeTeam");
CREATE INDEX ON match_raw ("AwayTeam");

-- =========================
-- CLEAN: טבלת משחקים מעובדים + פיצ'רים
-- ממפה קבוצות לשורות team_id ושומרת מינימום פיצ'רים שהקוד מחשב
-- =========================
CREATE TABLE match_clean (
  match_id               BIGSERIAL PRIMARY KEY,
  season                 TEXT NOT NULL,                 -- למשל '2023-24'
  match_date             DATE NOT NULL,

  home_team_id           INT NOT NULL REFERENCES team(team_id),
  away_team_id           INT NOT NULL REFERENCES team(team_id),

  home_goals             INT,                           -- מקביל ל-FTHG
  away_goals             INT,                           -- מקביל ל-FTAG
  result                 CHAR(1) CHECK (result IN ('H','D','A')),

  -- פיצ'רים שנגזרים בקוד:
  home_avg_conceded      REAL,                          -- ממוצע ספיגות בית
  away_avg_conceded      REAL,                          -- ממוצע ספיגות חוץ
  home_form              REAL,                          -- נק' ב-5 משחקים אחרונים
  away_form              REAL,                          -- נק' ב-5 משחקים אחרונים
  last5_matchup          REAL,                          -- נק' ב-5 מפגשים אחרונים H2H

  -- אופציונלי: עותק של נתוני הימורים וסטטיסטיקות שימושיות (כדי להימנע משאיבה מה-RAW)
  b365h                  NUMERIC(8,3),
  b365d                  NUMERIC(8,3),
  b365a                  NUMERIC(8,3),
  b365btsy               NUMERIC(8,3),
  b365btsn               NUMERIC(8,3),
  hs                     INT,
  away_shots             INT,                           -- Away shots (renamed from as)
  hc                     INT,
  ac                     INT,
  hy                     INT,
  ay                     INT,
  hr                     INT,
  ar                     INT,

  created_at             TIMESTAMPTZ DEFAULT now(),

  CONSTRAINT unique_game UNIQUE (season, match_date, home_team_id, away_team_id)
);

CREATE INDEX ON match_clean (match_date);
CREATE INDEX ON match_clean (home_team_id);
CREATE INDEX ON match_clean (away_team_id);

-- =========================
-- Fixtures: משחקים עתידיים מה-API (football-data.org)
-- =========================
CREATE TABLE fixture (
  fixture_id            BIGSERIAL PRIMARY KEY,
  provider_match_id     TEXT UNIQUE,                    -- id מה-API
  competition_code      TEXT DEFAULT 'PD',              -- ליגה ספרדית (ניתן לשינוי)
  matchday              INT,
  match_utc             TIMESTAMPTZ,
  home_team_id          INT REFERENCES team(team_id),
  away_team_id          INT REFERENCES team(team_id),
  status                TEXT,                           -- SCHEDULED / TIMED / וכו'
  inserted_at           TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON fixture (match_utc);
CREATE INDEX ON fixture (status);

-- =========================
-- Model registry: גרסת מודל + מטריקות + מיקום ארטיפקט
-- =========================
CREATE TABLE model_version (
  model_id       SERIAL PRIMARY KEY,
  model_name     TEXT NOT NULL,                         -- לדוגמה 'rf_v1' או 'poisson_v1'
  trained_on_dset TEXT,                                 -- תיאור/טווח תאריכים/האשים
  metrics        JSONB,                                 -- {"accuracy":..., "logloss":..., "brier":...}
  artifact_uri   TEXT,                                  -- נתיב לקובץ joblib / S3
  created_at     TIMESTAMPTZ DEFAULT now()
);

-- =========================
-- Predictions: תוצאות חיזוי למשחקי עתיד (Fixtures)
-- מבוסס על מה שהקוד מחזיר (בינארי H/A) אך מכין תשתית לשלוש תוצאות
-- =========================
CREATE TABLE prediction (
  pred_id        BIGSERIAL PRIMARY KEY,
  model_id       INT NOT NULL REFERENCES model_version(model_id),
  fixture_id     BIGINT NOT NULL REFERENCES fixture(fixture_id),

  -- הסתברויות. גם אם המודל הנוכחי בינארי (Home/Away), נשמור שלושתן כדי להתרחב בקלות.
  p_home         REAL CHECK (p_home >= 0 AND p_home <= 1),
  p_draw         REAL CHECK (p_draw >= 0 AND p_draw <= 1),
  p_away         REAL CHECK (p_away >= 0 AND p_away <= 1),

  -- xG (לא חובה בקוד, אבל הקוד מדבר על expected goals)
  expected_home_goals  REAL,
  expected_away_goals  REAL,

  -- צילום מהיר של פיצ'רים ששימשו (traceability/debug)
  feature_snapshot JSONB DEFAULT '{}'::jsonb,

  generated_at   TIMESTAMPTZ DEFAULT now(),

  CONSTRAINT unique_pred UNIQUE (model_id, fixture_id)
);

CREATE INDEX pred_by_fixture ON prediction (fixture_id);
CREATE INDEX pred_features_gin ON prediction USING GIN (feature_snapshot jsonb_path_ops);

-- =========================
-- קשרי עזר (לא חובה, אך מומלץ להרצות ולהגן מאי-עקביות)
-- =========================

-- 1) וידוא ששמות קבוצות ב-RAW קיימים בטבלת team (אפשר לממש טריגר/תהליך ETL שמבצע upsert לפני CLEAN)
--    כאן נשאר ברמת התהליך (Loader עושה upsert), לא אילוץ DB.

-- 2) שמירה על ערכי result תקינים
ALTER TABLE match_clean
  ADD CONSTRAINT result_valid CHECK (result IN ('H','D','A') OR result IS NULL);

-- 3) אם בוחרים להעתיק נתוני הימורים מה-RAW ל-CLEAN:
--    ניתן ליצור VIEW שמושך מה-RAW את ההימורים העדכניים לאותו משחק ולהצמיד ל-CLEAN, אם לא רוצים לשכפל נתונים.

-- =========================
-- תצוגה שימושית: משחקים בשבוע הקרוב עם תחזיות אחרונות
-- =========================
CREATE OR REPLACE VIEW v_upcoming_with_latest_pred AS
SELECT
  f.fixture_id,
  f.match_utc,
  th.name  AS home,
  ta.name  AS away,
  p.p_home,
  p.p_draw,
  p.p_away,
  p.generated_at,
  mv.model_name
FROM fixture f
JOIN team th ON th.team_id = f.home_team_id
JOIN team ta ON ta.team_id = f.away_team_id
LEFT JOIN LATERAL (
  SELECT p.*
  FROM prediction p
  WHERE p.fixture_id = f.fixture_id
  ORDER BY p.generated_at DESC
  LIMIT 1
) p ON TRUE
LEFT JOIN model_version mv ON mv.model_id = p.model_id
WHERE f.status = 'SCHEDULED';
