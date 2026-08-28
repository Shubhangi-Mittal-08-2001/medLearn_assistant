-- =====================================================
-- Medical Education Database Schema
-- Structured Clinical Data for RAG System Testing
-- =====================================================

-- Enable PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS pgvector;

-- =====================================================
-- Table 1: Somatosensory Receptors
-- Master reference table for all receptor types
-- =====================================================

DROP TABLE IF EXISTS somatosensory_receptors CASCADE;

CREATE TABLE somatosensory_receptors (
    receptor_id SERIAL PRIMARY KEY,
    receptor_name VARCHAR(100) NOT NULL UNIQUE,
    receptor_type VARCHAR(50) NOT NULL, -- 'mechanoreceptor', 'nociceptor', 'thermoreceptor', 'proprioceptor'
    adaptation_rate VARCHAR(20), -- 'rapid', 'slow', 'intermediate', 'N/A'
    receptive_field_size VARCHAR(20), -- 'small', 'large', 'variable'
    location_type VARCHAR(50), -- 'superficial', 'deep', 'muscle', 'joint', 'widespread'
    encapsulation BOOLEAN NOT NULL, -- TRUE if encapsulated, FALSE if free nerve ending
    primary_stimulus VARCHAR(100),
    response_characteristics TEXT,
    clinical_significance TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert receptor data
INSERT INTO somatosensory_receptors 
(receptor_name, receptor_type, adaptation_rate, receptive_field_size, location_type, encapsulation, primary_stimulus, response_characteristics, clinical_significance) 
VALUES
('Meissner Corpuscle', 'mechanoreceptor', 'rapid', 'small', 'superficial', TRUE, 
 'Light touch, vibration (30-50 Hz)', 
 'Detects fine vibrations and texture. Responds with brief burst during object movement. Essential for grip force adjustment.',
 'Loss in diabetic neuropathy and carpal tunnel syndrome. Critical for fine motor control.'),

('Merkel Receptor', 'mechanoreceptor', 'slow', 'small', 'superficial', FALSE,
 'Sustained pressure, edges, shapes',
 'Responds to initial indentation and sustained pressure up to several seconds. High density in digits (50/mm²).',
 'Density decreases with age (10/mm² by age 50). Essential for reading Braille and tactile discrimination.'),

('Pacinian Corpuscle', 'mechanoreceptor', 'rapid', 'large', 'deep', TRUE,
 'Deep pressure, vibration (250-350 Hz)',
 'Detects diffuse vibration and rapid pressure changes. Low threshold for high-frequency vibration.',
 'Used to detect tool vibration. Remains relatively preserved in aging compared to other receptors.'),

('Ruffini Corpuscle', 'mechanoreceptor', 'slow', 'large', 'deep', TRUE,
 'Skin stretch, sustained pressure',
 'Responds to lateral skin movement and stretching. Important for finger joint position sensing.',
 'Critical for proprioception in fingers. Damage affects grip control and object manipulation.'),

('Hair Follicle Receptor', 'mechanoreceptor', 'rapid', 'variable', 'superficial', FALSE,
 'Hair displacement, light touch',
 'Free nerve endings around hair follicles. Detect very light touch and movement across skin.',
 'Provides early warning of insects or objects contacting skin. Present only in hairy skin.'),

('Free Nerve Ending (Nociceptor)', 'nociceptor', 'N/A', 'variable', 'widespread', FALSE,
 'Noxious mechanical, thermal, chemical stimuli',
 'High-threshold mechanoreceptors and polymodal receptors. Respond to tissue damage signals.',
 'Mediate pain sensation. Dysfunction causes chronic pain syndromes or dangerous loss of pain sensation.'),

('Muscle Spindle', 'proprioceptor', 'both', 'variable', 'muscle', TRUE,
 'Muscle stretch and length changes',
 'Contains intrafusal fibers with Group Ia (rapid) and Group II (slow) afferents. Regulated by gamma motor neurons.',
 'Essential for proprioception and stretch reflexes. Damage impairs movement coordination and balance.'),

('Golgi Tendon Organ', 'proprioceptor', 'slow', 'variable', 'tendon', TRUE,
 'Muscle tension and force',
 'Located at muscle-tendon junction. Monitors muscle force to prevent injury from excessive contraction.',
 'Provides force feedback for motor control. Dysfunction can lead to muscle tears and coordination problems.'),

('Joint Receptor', 'proprioceptor', 'both', 'small', 'joint', TRUE,
 'Joint position and movement',
 'Low-threshold mechanoreceptors in joint capsules. Signal joint angle, movement direction, and speed.',
 'Essential for joint position sense. Arthritis and joint injuries impair proprioception.');

-- =====================================================
-- Table 2: Receptor Density by Body Location
-- Quantitative data on receptor distribution
-- =====================================================

DROP TABLE IF EXISTS receptor_density CASCADE;

CREATE TABLE receptor_density (
    density_id SERIAL PRIMARY KEY,
    receptor_id INTEGER REFERENCES somatosensory_receptors(receptor_id),
    body_location VARCHAR(100) NOT NULL,
    skin_type VARCHAR(20), -- 'glabrous', 'hairy'
    density_per_mm2 DECIMAL(6,2),
    age_group VARCHAR(20), -- 'young_adult', 'middle_age', 'elderly'
    age_range VARCHAR(20), -- e.g., '18-30', '31-50', '50+'
    notes TEXT
);

-- Insert density data
INSERT INTO receptor_density 
(receptor_id, body_location, skin_type, density_per_mm2, age_group, age_range, notes)
VALUES
-- Merkel receptors
(2, 'Fingertips', 'glabrous', 50.00, 'young_adult', '18-30', 'Highest density in digits for fine tactile discrimination'),
(2, 'Fingertips', 'glabrous', 30.00, 'middle_age', '31-50', 'Progressive decline in receptor density'),
(2, 'Fingertips', 'glabrous', 10.00, 'elderly', '50+', 'Significant reduction by age 50'),
(2, 'Palm', 'glabrous', 25.00, 'young_adult', '18-30', 'Lower than fingertips but still high'),
(2, 'Lips/Perioral', 'glabrous', 45.00, 'young_adult', '18-30', 'High density for speech and eating'),
(2, 'Forearm', 'hairy', 2.00, 'young_adult', '18-30', 'Very low density in hairy skin'),

-- Meissner corpuscles  
(1, 'Fingertips', 'glabrous', 40.00, 'young_adult', '18-30', 'Concentrated in dermal papillae'),
(1, 'Palm', 'glabrous', 20.00, 'young_adult', '18-30', 'Lower density than digits'),
(1, 'Sole of foot', 'glabrous', 15.00, 'young_adult', '18-30', 'Adapted for locomotion sensing'),

-- Pacinian corpuscles
(3, 'Fingertips', 'glabrous', 5.00, 'young_adult', '18-30', 'Deeper in dermis and subcutaneous tissue'),
(3, 'Palm', 'glabrous', 4.00, 'young_adult', '18-30', 'Distributed throughout hand'),
(3, 'Forearm', 'hairy', 3.00, 'young_adult', '18-30', 'Present in both hairy and glabrous skin'),

-- Ruffini corpuscles
(4, 'Fingertips', 'glabrous', 8.00, 'young_adult', '18-30', 'Important for finger proprioception'),
(4, 'Palm', 'glabrous', 6.00, 'young_adult', '18-30', 'Stretch sensors in hand'),

-- Hair follicle receptors
(5, 'Forearm', 'hairy', 20.00, 'young_adult', '18-30', 'One receptor per hair follicle'),
(5, 'Scalp', 'hairy', 15.00, 'young_adult', '18-30', 'Lower density than body hair');

-- =====================================================
-- Table 3: Pain Signal Types
-- Classification of pain pathways and characteristics
-- =====================================================

DROP TABLE IF EXISTS pain_signal_types CASCADE;

CREATE TABLE pain_signal_types (
    pain_type_id SERIAL PRIMARY KEY,
    pain_name VARCHAR(100) NOT NULL,
    alternate_name VARCHAR(100),
    fiber_type VARCHAR(20), -- 'A-delta', 'C-fiber', 'A-beta'
    conduction_velocity DECIMAL(5,1), -- meters per second
    transmission_speed VARCHAR(20), -- 'fast', 'slow', 'very_slow'
    spatial_resolution VARCHAR(20), -- 'high', 'low', 'moderate'
    localization VARCHAR(30), -- 'well_localized', 'poorly_localized'
    tolerability VARCHAR(30), -- 'easily_tolerated', 'poorly_tolerated'
    character_description TEXT,
    typical_duration VARCHAR(50),
    clinical_examples TEXT
);

-- Insert pain types
INSERT INTO pain_signal_types 
(pain_name, alternate_name, fiber_type, conduction_velocity, transmission_speed, spatial_resolution, localization, tolerability, character_description, typical_duration, clinical_examples)
VALUES
('First Pain', 'Cutaneous Pricking Pain', 'A-delta', 20.0, 'fast', 'high', 'well_localized', 'easily_tolerated',
 'Sharp, pricking sensation with rapid transmission and high spatial resolution. Allows precise localization of injury.',
 'Milliseconds to seconds',
 'Initial sharp pain from needle prick, paper cut, or acute injury. First sensation when touching hot object.'),

('Second Pain', 'Burning Pain', 'C-fiber', 1.0, 'slow', 'low', 'poorly_localized', 'poorly_tolerated',
 'Dull, burning, aching sensation that is highly affective and unpleasant. Delayed onset after injury.',
 'Seconds to minutes after injury',
 'Delayed burning pain after touching hot stove, sunburn pain, sustained aching after injury.'),

('Deep Pain', 'Visceral/Musculoskeletal Pain', 'C-fiber', 1.0, 'slow', 'low', 'poorly_localized', 'poorly_tolerated',
 'Arising from internal organs, muscles, and joints. Often described as deep, aching, cramping. May be chronic.',
 'Minutes to hours, can be chronic',
 'Heart attack (referred to arm), kidney stones, muscle strains, arthritis pain, menstrual cramps.');

-- =====================================================
-- Table 4: Drug Interactions with Pain Medications
-- Structured clinical reference data
-- =====================================================

DROP TABLE IF EXISTS drug_interactions CASCADE;

CREATE TABLE drug_interactions (
    interaction_id SERIAL PRIMARY KEY,
    drug_name VARCHAR(100) NOT NULL,
    drug_class VARCHAR(100),
    interacting_drug VARCHAR(100) NOT NULL,
    interacting_class VARCHAR(100),
    interaction_severity VARCHAR(20), -- 'major', 'moderate', 'minor'
    interaction_type VARCHAR(50), -- 'pharmacokinetic', 'pharmacodynamic', 'both'
    mechanism TEXT,
    clinical_effect TEXT,
    recommendation TEXT
);

-- Insert drug interaction data
INSERT INTO drug_interactions 
(drug_name, drug_class, interacting_drug, interacting_class, interaction_severity, interaction_type, mechanism, clinical_effect, recommendation)
VALUES
('Warfarin', 'Anticoagulant', 'NSAIDs (Ibuprofen, Naproxen)', 'Anti-inflammatory', 'major', 'pharmacodynamic',
 'Both drugs affect hemostasis. NSAIDs inhibit platelet function and may cause GI bleeding.',
 'Increased risk of bleeding, especially gastrointestinal hemorrhage. Risk of major bleeding events.',
 'Avoid combination if possible. If necessary, use lowest NSAID dose for shortest duration. Monitor INR closely and assess for bleeding.'),

('Warfarin', 'Anticoagulant', 'Acetaminophen', 'Analgesic', 'moderate', 'pharmacokinetic',
 'High doses of acetaminophen (>2g/day for several days) may enhance warfarin anticoagulant effect.',
 'Increased INR and bleeding risk with prolonged use of regular-strength acetaminophen.',
 'Monitor INR if acetaminophen used regularly. Consider dose reduction of acetaminophen or warfarin.'),

('Gabapentin', 'Anticonvulsant/Neuropathic pain', 'Opioids (Morphine, Oxycodone)', 'Analgesic', 'major', 'pharmacodynamic',
 'Additive CNS depression. Both drugs cause sedation and respiratory depression.',
 'Increased risk of severe sedation, respiratory depression, coma, and death.',
 'Avoid combination. If necessary, use lowest effective doses and monitor closely for respiratory depression.'),

('Tramadol', 'Opioid analgesic', 'SSRIs (Fluoxetine, Sertraline)', 'Antidepressant', 'major', 'pharmacodynamic',
 'Both drugs increase serotonin levels. Risk of serotonin syndrome.',
 'Serotonin syndrome: agitation, hallucinations, rapid heartbeat, fever, muscle rigidity, seizures.',
 'Avoid combination. If necessary, start with low doses and monitor for serotonin syndrome symptoms.'),

('Morphine', 'Opioid analgesic', 'Benzodiazepines (Diazepam, Alprazolam)', 'Anxiolytic', 'major', 'pharmacodynamic',
 'Additive CNS and respiratory depression.',
 'Extreme sedation, respiratory depression, coma, death. Black box warning from FDA.',
 'Avoid combination. Reserve for patients with no alternatives. Limit doses and duration.'),

('Pregabalin', 'Neuropathic pain', 'Alcohol', 'CNS Depressant', 'moderate', 'pharmacodynamic',
 'Additive CNS depression and impairment of motor and cognitive function.',
 'Increased drowsiness, dizziness, concentration difficulties, and impaired motor coordination.',
 'Advise patients to avoid alcohol. Warn about increased impairment if alcohol consumed.');

-- =====================================================
-- Table 5: Laboratory Reference Ranges
-- Normal values for diagnostic tests
-- =====================================================

DROP TABLE IF EXISTS lab_reference_ranges CASCADE;

CREATE TABLE lab_reference_ranges (
    lab_id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) NOT NULL,
    test_category VARCHAR(50),
    reference_range_min DECIMAL(10,3),
    reference_range_max DECIMAL(10,3),
    unit VARCHAR(50),
    age_group VARCHAR(30),
    sex VARCHAR(20), -- 'male', 'female', 'both'
    clinical_significance_low TEXT,
    clinical_significance_high TEXT,
    common_causes_abnormal TEXT
);

-- Insert lab reference ranges
INSERT INTO lab_reference_ranges 
(test_name, test_category, reference_range_min, reference_range_max, unit, age_group, sex, clinical_significance_low, clinical_significance_high, common_causes_abnormal)
VALUES
('Hemoglobin', 'Hematology', 13.5, 17.5, 'g/dL', 'adult', 'male',
 'Anemia - reduced oxygen carrying capacity, fatigue, weakness, pallor',
 'Polycythemia - increased blood viscosity, risk of thrombosis',
 'Low: iron deficiency, bleeding, chronic disease, B12/folate deficiency. High: dehydration, COPD, polycythemia vera'),

('Hemoglobin', 'Hematology', 12.0, 15.5, 'g/dL', 'adult', 'female',
 'Anemia - reduced oxygen carrying capacity, fatigue, weakness, pallor',
 'Polycythemia - increased blood viscosity, risk of thrombosis',
 'Low: iron deficiency, menstruation, pregnancy, chronic disease. High: dehydration, smoking, living at high altitude'),

('Platelet Count', 'Hematology', 150.0, 400.0, 'x10^9/L', 'adult', 'both',
 'Thrombocytopenia - increased bleeding risk, petechiae, bruising',
 'Thrombocytosis - increased clotting risk, may be reactive',
 'Low: ITP, leukemia, medication, sepsis. High: inflammation, iron deficiency, malignancy, post-splenectomy'),

('Glucose (Fasting)', 'Chemistry', 70.0, 99.0, 'mg/dL', 'adult', 'both',
 'Hypoglycemia - confusion, sweating, tremors, loss of consciousness',
 'Hyperglycemia - diabetes, increased infection risk, long-term complications',
 'Low: insulin excess, insulinoma, adrenal insufficiency. High: diabetes mellitus, steroid use, stress'),

('Creatinine', 'Chemistry', 0.7, 1.3, 'mg/dL', 'adult', 'male',
 'May indicate decreased muscle mass or malnutrition',
 'Kidney dysfunction, reduced GFR, potential need for dose adjustment of medications',
 'High: acute kidney injury, chronic kidney disease, dehydration, rhabdomyolysis, certain medications'),

('Creatinine', 'Chemistry', 0.6, 1.1, 'mg/dL', 'adult', 'female',
 'May indicate decreased muscle mass or malnutrition',
 'Kidney dysfunction, reduced GFR, potential need for dose adjustment of medications',
 'High: acute kidney injury, chronic kidney disease, dehydration, medications (NSAIDs, ACE inhibitors)'),

('INR (International Normalized Ratio)', 'Coagulation', 0.8, 1.2, 'ratio', 'adult', 'both',
 'Increased clotting risk if below therapeutic range on warfarin',
 'Bleeding risk - may indicate excessive anticoagulation or liver disease',
 'High: warfarin therapy (therapeutic 2-3), liver disease, vitamin K deficiency, DIC'),

('Vitamin B12', 'Vitamins', 200.0, 900.0, 'pg/mL', 'adult', 'both',
 'Pernicious anemia, peripheral neuropathy, cognitive impairment, fatigue',
 'Rarely clinically significant unless extremely elevated',
 'Low: pernicious anemia, malabsorption, vegan diet, metformin use, gastric surgery');

-- =====================================================
-- Table 6: Clinical Conditions Related to Receptors
-- Disease states affecting somatosensory system
-- =====================================================

DROP TABLE IF EXISTS clinical_conditions CASCADE;

CREATE TABLE clinical_conditions (
    condition_id SERIAL PRIMARY KEY,
    condition_name VARCHAR(100) NOT NULL,
    condition_category VARCHAR(50), -- 'neuropathy', 'compression', 'degenerative', 'inflammatory'
    affected_receptors TEXT[], -- Array of affected receptor types
    pathophysiology TEXT,
    clinical_presentation TEXT,
    diagnostic_criteria TEXT,
    treatment_approach TEXT,
    prognosis TEXT
);

-- Insert clinical conditions
INSERT INTO clinical_conditions 
(condition_name, condition_category, affected_receptors, pathophysiology, clinical_presentation, diagnostic_criteria, treatment_approach, prognosis)
VALUES
('Diabetic Peripheral Neuropathy', 'neuropathy', 
 ARRAY['Merkel Receptor', 'Meissner Corpuscle', 'Free Nerve Ending (Nociceptor)', 'Pacinian Corpuscle'],
 'Chronic hyperglycemia causes nerve damage through multiple mechanisms: increased polyol pathway flux, oxidative stress, advanced glycation end-products, and microvascular insufficiency. Progressive loss of receptor function and nerve fiber degeneration.',
 'Symmetrical sensory loss starting in feet (stocking-glove pattern). Loss of vibration sense, light touch, and pain sensation. Increased risk of foot ulcers and injuries. May progress to motor involvement.',
 'Symptoms consistent with neuropathy in patient with diabetes. Abnormal monofilament test, reduced vibration sensation (128 Hz tuning fork), decreased ankle reflexes. Nerve conduction studies show reduced velocities.',
 'Glycemic control is primary prevention. Pain management with gabapentin, pregabalin, or duloxetine. Foot care education critical. Alpha-lipoic acid may provide benefit.',
 'Progressive condition. Good glycemic control slows progression but rarely reverses damage. 50% of diabetics develop some neuropathy within 10 years.'),

('Carpal Tunnel Syndrome', 'compression',
 ARRAY['Meissner Corpuscle', 'Merkel Receptor', 'Pacinian Corpuscle'],
 'Compression of median nerve at wrist within carpal tunnel. Increased pressure damages nerve and reduces blood flow. Affects mechanoreceptor function in median nerve distribution (thumb, index, middle, and radial half of ring finger).',
 'Numbness, tingling, and pain in median nerve distribution. Worse at night. Thenar muscle weakness and atrophy in advanced cases. Positive Phalen and Tinel signs.',
 'Clinical symptoms in median nerve distribution. Positive electrodiagnostic studies showing median nerve slowing across wrist. Tinel sign at wrist, Phalen maneuver reproduces symptoms.',
 'Conservative: wrist splinting (especially at night), activity modification, NSAIDs. Corticosteroid injection for moderate cases. Surgical carpal tunnel release for severe or refractory cases.',
 'Excellent with early treatment. Surgical release has 90%+ success rate. Delayed treatment may result in permanent nerve damage and thenar atrophy.'),

('Age-Related Sensory Decline', 'degenerative',
 ARRAY['Merkel Receptor', 'Meissner Corpuscle', 'Pacinian Corpuscle', 'Muscle Spindle'],
 'Progressive loss of receptor density with aging. Merkel receptor density decreases from 50/mm² to 10/mm² by age 50. Reduced nerve fiber density, slower nerve conduction, decreased CNS processing.',
 'Reduced tactile acuity, decreased vibration sense, impaired proprioception. Increased fall risk. Difficulty with fine motor tasks like buttoning clothes. Reduced ability to detect temperature changes.',
 'Age-related decline in sensory testing. Reduced two-point discrimination, decreased vibration sense (particularly in lower extremities), impaired joint position sense.',
 'No cure. Balance training and fall prevention programs. Adequate lighting. Regular vision and hearing checks. Assistive devices as needed. Medication review to minimize drugs affecting balance.',
 'Progressive but highly variable. Some individuals maintain good sensory function into advanced age. Fall prevention crucial as falls are major cause of morbidity in elderly.'),

('Complex Regional Pain Syndrome (CRPS)', 'inflammatory',
 ARRAY['Free Nerve Ending (Nociceptor)', 'All mechanoreceptors'],
 'Abnormal sympathetic nervous system response to injury. Peripheral and central sensitization. Neurogenic inflammation with plasma extravasation and edema. Altered pain processing and receptor sensitization.',
 'Severe burning pain disproportionate to injury. Allodynia (pain from non-painful stimuli). Swelling, skin color/temperature changes. Motor dysfunction and tremor. Progresses through stages.',
 'Budapest criteria: Continuing pain disproportionate to inciting event, plus symptoms in 3 of 4 categories (sensory, vasomotor, sudomotor/edema, motor/trophic) and signs in 2+ categories.',
 'Early aggressive treatment crucial. Physical therapy (desensitization, range of motion). Pain management with gabapentin, topical agents. Sympathetic blocks for some patients. Psychological support.',
 'Better with early treatment. Stage 1 (acute): potentially reversible. Stage 2-3: may develop permanent changes. Multidisciplinary approach improves outcomes.');

-- =====================================================
-- Create indexes for better query performance
-- =====================================================

CREATE INDEX idx_receptors_type ON somatosensory_receptors(receptor_type);
CREATE INDEX idx_receptors_adaptation ON somatosensory_receptors(adaptation_rate);
CREATE INDEX idx_density_location ON receptor_density(body_location);
CREATE INDEX idx_density_receptor ON receptor_density(receptor_id);
CREATE INDEX idx_drugs_severity ON drug_interactions(interaction_severity);
CREATE INDEX idx_drugs_name ON drug_interactions(drug_name);
CREATE INDEX idx_labs_category ON lab_reference_ranges(test_category);
CREATE INDEX idx_conditions_category ON clinical_conditions(condition_category);

-- =====================================================
-- Create views for common queries
-- =====================================================

-- View: Rapidly vs Slowly Adapting Receptors Comparison
CREATE OR REPLACE VIEW receptor_adaptation_comparison AS
SELECT 
    adaptation_rate,
    COUNT(*) as receptor_count,
    STRING_AGG(receptor_name, ', ' ORDER BY receptor_name) as receptors,
    STRING_AGG(DISTINCT location_type, ', ') as locations
FROM somatosensory_receptors
WHERE adaptation_rate IN ('rapid', 'slow')
GROUP BY adaptation_rate;

-- View: High-Risk Drug Interactions  
CREATE OR REPLACE VIEW high_risk_drug_interactions AS
SELECT 
    drug_name,
    drug_class,
    interacting_drug,
    interacting_class,
    clinical_effect,
    recommendation
FROM drug_interactions
WHERE interaction_severity = 'major'
ORDER BY drug_name;

-- View: Receptor Density by Age
CREATE OR REPLACE VIEW receptor_density_age_comparison AS
SELECT 
    sr.receptor_name,
    rd.body_location,
    rd.age_group,
    rd.density_per_mm2,
    ROUND(
        (rd.density_per_mm2 / FIRST_VALUE(rd.density_per_mm2) 
         OVER (PARTITION BY sr.receptor_id, rd.body_location ORDER BY rd.age_group)) * 100, 
        1
    ) as percentage_of_young_adult
FROM receptor_density rd
JOIN somatosensory_receptors sr ON rd.receptor_id = sr.receptor_id
ORDER BY sr.receptor_name, rd.body_location, rd.age_group;

-- =====================================================
-- Sample Queries for Testing
-- =====================================================

-- Query 1: Find all mechanoreceptors and their characteristics
-- SELECT receptor_name, adaptation_rate, receptive_field_size, primary_stimulus 
-- FROM somatosensory_receptors 
-- WHERE receptor_type = 'mechanoreceptor';

-- Query 2: Compare receptor density in fingertips across age groups
-- SELECT age_group, age_range, density_per_mm2 
-- FROM receptor_density 
-- WHERE body_location = 'Fingertips' AND receptor_id = 2
-- ORDER BY density_per_mm2 DESC;

-- Query 3: Find all major drug interactions with warfarin
-- SELECT interacting_drug, clinical_effect, recommendation 
-- FROM drug_interactions 
-- WHERE drug_name = 'Warfarin' AND interaction_severity = 'major';

-- Query 4: Get pain signal characteristics
-- SELECT pain_name, fiber_type, transmission_speed, localization, character_description 
-- FROM pain_signal_types;

-- Query 5: Find conditions affecting Merkel receptors
-- SELECT condition_name, pathophysiology, clinical_presentation 
-- FROM clinical_conditions 
-- WHERE 'Merkel Receptor' = ANY(affected_receptors);

COMMENT ON TABLE somatosensory_receptors IS 'Master reference table for all types of somatosensory receptors including mechanoreceptors, nociceptors, thermoreceptors, and proprioceptors';
COMMENT ON TABLE receptor_density IS 'Quantitative data on receptor distribution across different body locations and age groups';
COMMENT ON TABLE pain_signal_types IS 'Classification of pain pathways with transmission characteristics and clinical correlations';
COMMENT ON TABLE drug_interactions IS 'Structured reference for drug-drug interactions relevant to pain management';
COMMENT ON TABLE lab_reference_ranges IS 'Normal laboratory values and clinical significance of abnormal results';
COMMENT ON TABLE clinical_conditions IS 'Disease states affecting the somatosensory system with diagnostic and treatment information';
