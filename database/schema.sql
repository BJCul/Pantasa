-- SQL script to create necessary database tables
CREATE TABLE sentences (
    id SERIAL PRIMARY KEY,
    original_sentence TEXT NOT NULL,
    corrected_sentence TEXT NOT NULL
);
