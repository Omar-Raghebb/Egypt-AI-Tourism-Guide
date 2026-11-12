# Sample RAG outputs (captured locally, fallback mode — no GROQ_API_KEY)

- Pipeline health: `{"kb_loaded": true, "kb_places": 6, "kb_faqs": 96, "chroma_docs": 212, "bm25_docs": 212, "embedding_model": "intfloat/multilingual-e5-large", "groq_configured": false, "groq_model": "openai/gpt-oss-120b"}`
- `generated_by=fallback` means Groq was not called because no API key is
  configured; the text after the notice is the exact retrieved evidence.

## Query (en): What is the ticket price of the Pyramids of Giza?

- intent: `price` - methods: `['dense', 'sparse', 'structured', 'structured_fields']` - entities: `{"places": ["giza_pyramids"], "city": "Giza", "structured_fields": ["ticket_price"]}` - latency: 19241 ms
- verification_flags: `[]`

```text
⚠️ Generation is temporarily unavailable (the LLM provider could not be reached). Here is the information retrieved from the knowledge base:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### STRUCTURED RECORD — Pyramids of Giza (place_id: giza_pyramids)
verification: verified (last_verified: 2026-07)
[giza_pyramids] Pyramids of Giza (أهرامات الجيزة)
City: Giza
Category: Ancient Monument / Necropolis / UNESCO World Heritage Site
Coordinates: lat 29.9792, lon 31.1342
Opening hours: 07:00–19:00 daily (Giza Plateau general entry); 08:00–17:00 daily (Giza Plateau general entry)
Ticket prices (verified): general plateau entry foreign adult: 700 EGP; great pyramid khufu interior foreign adult: 1500 EGP; great pyramid khufu interior foreign student: 750 EGP; egyptian national general entry: 150 EGP; egyptian student general entry: 75 EGP
accessibility_notes: Uneven sandy and rocky terrain; wheelchair access is limited on the plateau itself, though a paved viewing area is available. Pyramid interiors involve steep, low-ceilinged passages and are not wheelchair accessible.
nearest_airport: Cairo International Airport (CAI), approx. 30–40 km away

### DOCUMENT giza_pyramids::practical::0
place_id: giza_pyramids | section: practical | verification: verified | source: giza_pyramids/info.json
{
  "opening_hours": {
    "summer_may_to_september": "07:00–19:00 daily (Giza Plateau general entry)",
    "winter_october_to_april": "08:00–17:00 daily (Giza Plateau general entry)",
    "interior_pyramid_access": "Limited daily visitor quota for entering the Great Pyramid's interior chambers; last entry typically 1 hour before closing",
    "notes": "Hours can shift slightly for religious holidays (Ramadan, Eid) and are set by the Ministry of Tourism and Antiquities"
  },
  "ticket_price": {
    "currency": "EGP",
    "general_plateau_entry_foreign_adult": 700,
    "great_pyramid_khufu_interior_foreign_adult": 1500,
    "great_pyramid_khufu_interior_foreign_student": 750,
    "egyptian_national_general_entry": 150,
    "egyptian_student_general_entry": 75,
    "sphinx_viewing": "included in general plateau entry",
    "notes": "Prices are set by Egypt's Supreme Council of Antiquities and are revised periodically (last major revision effective January 2025). Interior access to Khafre's and Menkaure's pyramids is ticketed separately and subject to availability. Always confirm current prices via the official egymonuments.com portal or at the gate before visiting.",
    "last_verified": "2026-07"
  },
  "best_visiting_time": {
    "time_of_day": "Early morning (opening time) or 1-2 hours before closing, to avoid heat and tour bus crowds",
    "season": "October to April (cooler, more comfortable daytime temperatures)",
    "avoid": "Midday in June–August, when temperatures often exceed 38°C (100°F)"
  }
}

### DOCUMENT giza_pyramids::faq_8
place_id: giza_pyramids | section: faq | verification: verified | source: giza_pyramids/info.json
Q: Is the Great Sphinx included in the pyramid ticket?
A: Yes, viewing the Great Sphinx is included in the general Giza Plateau entrance ticket; no separate ticket is required.

### DOCUMENT giza_pyramids::faq_2
place_id: giza_pyramids | section: faq | verification: verified | source: giza_pyramids/info.json
Q: What are the opening hours of the Giza Pyramids?
A: The plateau is generally open from 8:00 am to 5:00 pm in winter (October–April) and 7:00 am to 7:00 pm in summer (May–September), though hours can vary slightly around holidays.

### DOCUMENT giza_pyramids::faq_0::0
place_id: giza_pyramids | section: faq | verification: verified | source: giza_pyramids/info.json
Q: How much does it cost to enter the Pyramids of Giza?
A: General entry to the Giza Plateau (which includes viewing all three pyramids and the Great Sphinx from outside) costs 700 EGP for foreign adults. Entering inside the Great Pyramid of Khufu costs an additional 1,500 EGP for foreign adults. Egyptian nationals pay significantly reduced local rates

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```

## Query (ar): سعر تذكرة أهرامات الجيزة؟

- intent: `price` - methods: `['dense', 'sparse', 'structured', 'structured_fields']` - entities: `{"places": ["giza_pyramids"], "city": "Giza", "structured_fields": ["ticket_price"]}` - latency: 35 ms
- verification_flags: `[]`

```text
⚠️ ميزة التوليد غير متوفرة حالياً (تعذر الاتصال بخدمة الذكاء الاصطناعي). هذه هي المعلومات المسترجعة من قاعدة المعرفة:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### STRUCTURED RECORD — Pyramids of Giza (place_id: giza_pyramids)
verification: verified (last_verified: 2026-07)
[giza_pyramids] Pyramids of Giza (أهرامات الجيزة)
City: Giza
Category: Ancient Monument / Necropolis / UNESCO World Heritage Site
Coordinates: lat 29.9792, lon 31.1342
Opening hours: 07:00–19:00 daily (Giza Plateau general entry); 08:00–17:00 daily (Giza Plateau general entry)
Ticket prices (verified): general plateau entry foreign adult: 700 EGP; great pyramid khufu interior foreign adult: 1500 EGP; great pyramid khufu interior foreign student: 750 EGP; egyptian national general entry: 150 EGP; egyptian student general entry: 75 EGP
accessibility_notes: Uneven sandy and rocky terrain; wheelchair access is limited on the plateau itself, though a paved viewing area is available. Pyramid interiors involve steep, low-ceilinged passages and are not wheelchair accessible.
nearest_airport: Cairo International Airport (CAI), approx. 30–40 km away

### DOCUMENT giza_pyramids::practical::0
place_id: giza_pyramids | section: practical | verification: verified | source: giza_pyramids/info.json
{
  "opening_hours": {
    "summer_may_to_september": "07:00–19:00 daily (Giza Plateau general entry)",
    "winter_october_to_april": "08:00–17:00 daily (Giza Plateau general entry)",
    "interior_pyramid_access": "Limited daily visitor quota for entering the Great Pyramid's interior chambers; last entry typically 1 hour before closing",
    "notes": "Hours can shift slightly for religious holidays (Ramadan, Eid) and are set by the Ministry of Tourism and Antiquities"
  },
  "ticket_price": {
    "currency": "EGP",
    "general_plateau_entry_foreign_adult": 700,
    "great_pyramid_khufu_interior_foreign_adult": 1500,
    "great_pyramid_khufu_interior_foreign_student": 750,
    "egyptian_national_general_entry": 150,
    "egyptian_student_general_entry": 75,
    "sphinx_viewing": "included in general plateau entry",
    "notes": "Prices are set by Egypt's Supreme Council of Antiquities and are revised periodically (last major revision effective January 2025). Interior access to Khafre's and Menkaure's pyramids is ticketed separately and subject to availability. Always confirm current prices via the official egymonuments.com portal or at the gate before visiting.",
    "last_verified": "2026-07"
  },
  "best_visiting_time": {
    "time_of_day": "Early morning (opening time) or 1-2 hours before closing, to avoid heat and tour bus crowds",
    "season": "October to April (cooler, more comfortable daytime temperatures)",
    "avoid": "Midday in June–August, when temperatures often exceed 38°C (100°F)"
  }
}

### DOCUMENT giza_pyramids::overview::0
place_id: giza_pyramids | section: overview | verification: verified | source: giza_pyramids/info.json
Pyramids of Giza (أهرامات الجيزة)
Category: Ancient Monument / Necropolis / UNESCO World Heritage Site
City: Giza
Region: Greater Cairo
Country: Egypt
Coordinates: latitude 29.9792, longitude 31.1342

### DOCUMENT grand_egyptian_museum::practical::0
place_id: grand_egyptian_museum | section: practical | verification: verified | source: grand_egyptian_museum/info.json
{
  "opening_hours": {
    "typical_hours": "09:00–19:00 (hours and days can vary seasonally; night hours sometimes extended for peak season)",
    "closed_days": "Varies by season; check official channels before visiting",
    "booking_requirement": "Timed, date-specific entry tickets must be booked in advance online; walk-up/on-site ticket sales are generally not available"
  },
  "ticket_price": {
    "currency": "EGP",
    "foreign_adult": 1450,
    "foreign_child_student_6_to_25": 730,
    "egyptian_national_adult": 200,
    "children_under_6": "free",
    "notes": "Prices rose significantly after the November 2025 full opening compared to earlier soft-opening rates (some older sources cite EGP 500-600, which are outdated). Tickets are sold only through the official portal (visit-gem.com) and are non-refundable and non-transferable once booked. Beware of unofficial or fraudulent ticketing websites.",
    "last_verified": "2026-07"
  },
  "best_visiting_time": {
    "time_of_day": "Morning slots are less crowded; late-afternoon slots allow combining with sunset views of the pyramids from museum terraces",
    "season": "Year-round (indoor, climate-controlled), though cooler months (October–April) pair better with an outdoor pyramid visit the same day",
    "duration_recommended": "3 to 6 hours depending on depth of interest"
  }
}

### DOCUMENT giza_pyramids::faq_0::0
place_id: giza_pyramids | section: faq | verification: verified | source: giza_pyramids/info.json
Q: How much does it cost to enter the Pyramids of Giza?
A: General entry to the Giza Plateau (which includes viewing all three pyramids and the Great Sphinx from outside) costs 700 EGP for foreign adults. Entering inside the Great Pyramid of Khufu costs an additional 1,500 EGP for foreign adults. Egyptian nationals pay significantly reduced local rates

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```

## Query (en): What are the opening hours of the Pyramids of Giza?

- intent: `hours` - methods: `['dense', 'sparse', 'structured', 'structured_fields']` - entities: `{"places": ["giza_pyramids"], "city": "Giza", "structured_fields": ["opening_hours"]}` - latency: 29 ms
- verification_flags: `[]`

```text
⚠️ Generation is temporarily unavailable (the LLM provider could not be reached). Here is the information retrieved from the knowledge base:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### STRUCTURED RECORD — Pyramids of Giza (place_id: giza_pyramids)
verification: verified (last_verified: 2026-07)
[giza_pyramids] Pyramids of Giza (أهرامات الجيزة)
City: Giza
Category: Ancient Monument / Necropolis / UNESCO World Heritage Site
Coordinates: lat 29.9792, lon 31.1342
Opening hours: 07:00–19:00 daily (Giza Plateau general entry); 08:00–17:00 daily (Giza Plateau general entry)
Ticket prices (verified): general plateau entry foreign adult: 700 EGP; great pyramid khufu interior foreign adult: 1500 EGP; great pyramid khufu interior foreign student: 750 EGP; egyptian national general entry: 150 EGP; egyptian student general entry: 75 EGP
accessibility_notes: Uneven sandy and rocky terrain; wheelchair access is limited on the plateau itself, though a paved viewing area is available. Pyramid interiors involve steep, low-ceilinged passages and are not wheelchair accessible.
nearest_airport: Cairo International Airport (CAI), approx. 30–40 km away

### DOCUMENT giza_pyramids::practical::0
place_id: giza_pyramids | section: practical | verification: verified | source: giza_pyramids/info.json
{
  "opening_hours": {
    "summer_may_to_september": "07:00–19:00 daily (Giza Plateau general entry)",
    "winter_october_to_april": "08:00–17:00 daily (Giza Plateau general entry)",
    "interior_pyramid_access": "Limited daily visitor quota for entering the Great Pyramid's interior chambers; last entry typically 1 hour before closing",
    "notes": "Hours can shift slightly for religious holidays (Ramadan, Eid) and are set by the Ministry of Tourism and Antiquities"
  },
  "ticket_price": {
    "currency": "EGP",
    "general_plateau_entry_foreign_adult": 700,
    "great_pyramid_khufu_interior_foreign_adult": 1500,
    "great_pyramid_khufu_interior_foreign_student": 750,
    "egyptian_national_general_entry": 150,
    "egyptian_student_general_entry": 75,
    "sphinx_viewing": "included in general plateau entry",
    "notes": "Prices are set by Egypt's Supreme Council of Antiquities and are revised periodically (last major revision effective January 2025). Interior access to Khafre's and Menkaure's pyramids is ticketed separately and subject to availability. Always confirm current prices via the official egymonuments.com portal or at the gate before visiting.",
    "last_verified": "2026-07"
  },
  "best_visiting_time": {
    "time_of_day": "Early morning (opening time) or 1-2 hours before closing, to avoid heat and tour bus crowds",
    "season": "October to April (cooler, more comfortable daytime temperatures)",
    "avoid": "Midday in June–August, when temperatures often exceed 38°C (100°F)"
  }
}

### DOCUMENT giza_pyramids::faq_2
place_id: giza_pyramids | section: faq | verification: verified | source: giza_pyramids/info.json
Q: What are the opening hours of the Giza Pyramids?
A: The plateau is generally open from 8:00 am to 5:00 pm in winter (October–April) and 7:00 am to 7:00 pm in summer (May–September), though hours can vary slightly around holidays.

### DOCUMENT giza_pyramids::faq_7
place_id: giza_pyramids | section: faq | verification: verified | source: giza_pyramids/info.json
Q: What is the best time of day to visit the pyramids?
A: Early morning right at opening, or the last one to two hours before closing, offer cooler temperatures, softer light for photography, and smaller crowds compared to midday.

### DOCUMENT giza_pyramids::overview::1
place_id: giza_pyramids | section: overview | verification: verified | source: giza_pyramids/info.json
Overview: The Pyramids of Giza form the last surviving wonder of the ancient Seven Wonders of the World

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```

## Query (ar): ما هي مواعيد فتح معبد الأقصر؟

- intent: `hours` - methods: `['dense', 'sparse', 'structured', 'structured_fields']` - entities: `{"places": ["luxor_temple"], "city": "Luxor", "structured_fields": ["opening_hours"]}` - latency: 31 ms
- verification_flags: `[]`

```text
⚠️ ميزة التوليد غير متوفرة حالياً (تعذر الاتصال بخدمة الذكاء الاصطناعي). هذه هي المعلومات المسترجعة من قاعدة المعرفة:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### STRUCTURED RECORD — Luxor Temple (place_id: luxor_temple)
verification: verified (last_verified: 2026-07)
[luxor_temple] Luxor Temple (معبد الأقصر)
City: Luxor
Category: Ancient Temple / UNESCO World Heritage Site
Coordinates: lat 25.6997, lon 32.6396
Opening hours: {"daily_hours": "06:00–19:00 (approximate; often the last temple visited in the evening due to dramatic floodlighting)", "notes": "Luxor Temple is popular for evening visits when floodlights illuminate the columns and statues dramatically"}
Ticket prices (verified): foreign adult: 500 EGP; foreign student: 250 EGP; egyptian adult: 40 EGP; egyptian student: 20 EGP
accessibility_notes: Mostly paved, flat pathways within the temple courtyards, making it one of the more accessible ancient sites in Luxor, though some interior areas have uneven ancient stone flooring.
nearest_airport: Luxor International Airport (LXR), approx. 7 km away

### DOCUMENT luxor_temple::practical::0
place_id: luxor_temple | section: practical | verification: verified | source: luxor_temple/info.json
{
  "opening_hours": {
    "daily_hours": "06:00–19:00 (approximate; often the last temple visited in the evening due to dramatic floodlighting)",
    "notes": "Luxor Temple is popular for evening visits when floodlights illuminate the columns and statues dramatically"
  },
  "ticket_price": {
    "currency": "EGP",
    "foreign_adult": 500,
    "foreign_student": 250,
    "egyptian_adult": 40,
    "egyptian_student": 20,
    "notes": "Prices are set by the Ministry of Tourism and Antiquities and subject to periodic revision; confirm current rates via egymonuments.com or at the ticket office.",
    "last_verified": "2026-07"
  },
  "best_visiting_time": {
    "time_of_day": "Evening, after sunset, when the temple is dramatically floodlit and daytime heat has subsided",
    "season": "October to April for milder daytime weather if visiting earlier in the day",
    "note": "Luxor Temple is one of the few major Egyptian sites where a nighttime visit is often considered superior to a daytime one"
  }
}

### DOCUMENT luxor_temple::overview::0
place_id: luxor_temple | section: overview | verification: verified | source: luxor_temple/info.json
Luxor Temple (معبد الأقصر)
Category: Ancient Temple / UNESCO World Heritage Site
City: Luxor
Region: Upper Egypt (ancient Thebes, East Bank of the Nile, central Luxor city)
Country: Egypt
Coordinates: latitude 25.6997, longitude 32.6396

### DOCUMENT karnak_temple::overview::0
place_id: karnak_temple | section: overview | verification: verified | source: karnak_temple/info.json
Karnak Temple Complex (معبد الكرنك)
Category: Ancient Temple Complex / UNESCO World Heritage Site
City: Luxor
Region: Upper Egypt (ancient Thebes, East Bank of the Nile)
Country: Egypt
Coordinates: latitude 25.7188, longitude 32.6573

### DOCUMENT karnak_temple::practical::0
place_id: karnak_temple | section: practical | verification: verified | source: karnak_temple/info.json
{
  "opening_hours": {
    "daily_hours": "06:00–17:30 (October–April); 06:00–18:00 (May–September), approximate and subject to change",
    "sound_and_light_show": "Evening shows typically run at staggered times (e.g., 19:00, 20:15, 21:30) with rotating language narration"
  },
  "ticket_price": {
    "currency": "EGP",
    "foreign_adult": 600,
    "foreign_student": 300,
    "egyptian_adult": 40,
    "egyptian_student": 20,
    "sound_and_light_show_adult": 300,
    "notes": "Prices are set by the Ministry of Tourism and Antiquities and subject to periodic revision; confirm current rates via egymonuments.com or at the ticket office.",
    "last_verified": "2026-07"
  },
  "best_visiting_time": {
    "time_of_day": "Early morning at opening (cooler temperatures, smaller crowds) or late afternoon after large tour groups depart",
    "season": "October to April for milder weather",
    "avoid": "Midday hours (10:00–14:00), which combine peak heat and peak crowds",
    "evening_option": "The Sound and Light Show offers a dramatically lit nighttime experience of the temple"
  }
}

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```

## Query (en): What are the best places to visit in Luxor?

- intent: `recommendation` - methods: `['dense', 'sparse', 'structured']` - entities: `{"places": [], "city": "Luxor", "structured_fields": []}` - latency: 31 ms
- verification_flags: `[]`

```text
⚠️ Generation is temporarily unavailable (the LLM provider could not be reached). Here is the information retrieved from the knowledge base:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### STRUCTURED RECORD — Karnak Temple Complex (place_id: karnak_temple)
verification: verified (last_verified: 2026-07)
[karnak_temple] Karnak Temple Complex (معبد الكرنك)
City: Luxor
Category: Ancient Temple Complex / UNESCO World Heritage Site
Coordinates: lat 25.7188, lon 32.6573
Opening hours: {"daily_hours": "06:00–17:30 (October–April); 06:00–18:00 (May–September), approximate and subject to change", "sound_and_light_show": "Evening shows typically run at staggered times (e.g., 19:00, 20:15, 21:30) with rotating language narration"}
Ticket prices (verified): foreign adult: 600 EGP; foreign student: 300 EGP; egyptian adult: 40 EGP; egyptian student: 20 EGP; sound and light show adult: 300 EGP
accessibility_notes: Mostly flat paved and packed-earth paths suitable for most visitors, though some areas have uneven ancient stone flooring; limited wheelchair accessibility in certain inner sanctuaries.
nearest_airport: Luxor International Airport (LXR), approx. 8 km away

### STRUCTURED RECORD — Luxor Temple (place_id: luxor_temple)
verification: verified (last_verified: 2026-07)
[luxor_temple] Luxor Temple (معبد الأقصر)
City: Luxor
Category: Ancient Temple / UNESCO World Heritage Site
Coordinates: lat 25.6997, lon 32.6396
Opening hours: {"daily_hours": "06:00–19:00 (approximate; often the last temple visited in the evening due to dramatic floodlighting)", "notes": "Luxor Temple is popular for evening visits when floodlights illuminate the columns and statues dramatically"}
Ticket prices (verified): foreign adult: 500 EGP; foreign student: 250 EGP; egyptian adult: 40 EGP; egyptian student: 20 EGP
accessibility_notes: Mostly paved, flat pathways within the temple courtyards, making it one of the more accessible ancient sites in Luxor, though some interior areas have uneven ancient stone flooring.
nearest_airport: Luxor International Airport (LXR), approx. 7 km away

### STRUCTURED RECORD — Valley of the Kings (place_id: valley_of_the_kings)
verification: verified (last_verified: 2026-07)
[valley_of_the_kings] Valley of the Kings (وادي الملوك)
City: Luxor
Category: Ancient Necropolis / Royal Tomb Complex / UNESCO World Heritage Site
Coordinates: lat 25.7402, lon 32.6014
Opening hours: {"daily_hours": "06:00–17:00 (approximate, may extend slightly in summer months)", "notes": "Some tombs are rotated in and out of public access periodically for conservation purposes"}
Ticket prices (verified): general entry foreign adult: 750 EGP; general entry foreign student: 375 EGP; egyptian adult: 60 EGP; egyptian student: 30 EGP; tutankhamun tomb kv62 separate ticket foreign adult: 700 EGP; seti i tomb kv17 separate ticket foreign adult: 2000 EGP
accessibility_notes: Terrain is uneven desert ground; tomb interiors typically require descending ramps or steps into narrow corridors, making most tombs difficult or inaccessible for wheelchair users. A tram serves the walk from the parking area to the valley entrance.
nearest_airport: Luxor International Airport (LXR), approx. 20 km away

### DOCUMENT luxor_temple::practical::0
place_id: luxor_temple | section: practical | verification: verified | source: luxor_temple/info.json
{
  "opening_hours": {
    "daily_hours": "06:00–19:00 (approximate; often the last temple visited in the evening due to dramatic floodlighting)",
    "notes": "Luxor Temple is popular for evening visits when floodlights illuminate the columns and statues dramatically"
  },
  "ticket_price": {
    "currency": "EGP",
    "foreign_adult": 500,
    "foreign_student": 250,
    "egyptian_adult": 40,
    "egyptian_student": 20,
    "notes": "Prices are set by the Ministry of Tourism and Antiquities and subject to periodic revision; confirm current rates via egymonuments.com or at the ticket office.",
    "last_verified": "2026-07"
  },
  "best_visiting_time": {
    "time_of_day": "Evening, after sunset, when the temple is dramatically floodlit and daytime heat has subsided",
    "season": "October to April for milder daytime weather if visiting earlier in the day",
    "note": "Luxor Temple is one of the few major Egyptian sites where a nighttime visit is often considered superior to a daytime one"
  }
}

### DOCUMENT luxor_temple::faq_2
place_id: luxor_temple | section: faq | verification: verified | source: luxor_temple/info.json
Q: Is Luxor Temple better to visit during the day or at night?
A: Many visitors and guides consider Luxor Temple one of the best evening visits in Egypt. Floodlighting after sunset dramatically illuminates the columns, statues, and pylon, and daytime heat is no longer a factor.

### DOCUMENT karnak_temple::faq_8
place_id: karnak_temple | section: faq | verification: verified | source: karnak_temple/info.json
Q: What is the best time of day to visit Karnak?
A: Early morning right at opening offers cooler temperatures and fewer tour groups; late afternoon after large groups depart is another good window. Midday hours combine peak heat and crowds and are best avoided.

### DOCUMENT luxor_temple::accessibility::1
place_id: luxor_temple | section: accessibility | verification: verified | source: luxor_temple/info.json
Nearest airport: Luxor International Airport (LXR), approx. 7 km away

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```

## Query (en): What are the exact coordinates of the Giza Pyramids?

- intent: `coordinates` - methods: `['dense', 'sparse']` - entities: `{"places": [], "city": "Giza", "structured_fields": ["location.coordinates"]}` - latency: 39 ms
- verification_flags: `["the query asks for exact structured data but no specific place could be identified; exact values cannot be given without a place"]`

```text
⚠️ Generation is temporarily unavailable (the LLM provider could not be reached). Here is the information retrieved from the knowledge base:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### DOCUMENT giza_pyramids::overview::0
place_id: giza_pyramids | section: overview | verification: verified | source: giza_pyramids/info.json
Pyramids of Giza (أهرامات الجيزة)
Category: Ancient Monument / Necropolis / UNESCO World Heritage Site
City: Giza
Region: Greater Cairo
Country: Egypt
Coordinates: latitude 29.9792, longitude 31.1342

### DOCUMENT giza_pyramids::overview::1
place_id: giza_pyramids | section: overview | verification: verified | source: giza_pyramids/info.json
Overview: The Pyramids of Giza form the last surviving wonder of the ancient Seven Wonders of the World

### DOCUMENT grand_egyptian_museum::overview::0
place_id: grand_egyptian_museum | section: overview | verification: verified | source: grand_egyptian_museum/info.json
Grand Egyptian Museum (GEM) (المتحف المصري الكبير)
Category: Museum
City: Giza
Region: Greater Cairo, adjacent to the Giza Plateau
Country: Egypt
Coordinates: latitude 29.9925, longitude 31.1131

### DOCUMENT grand_egyptian_museum::faq_4
place_id: grand_egyptian_museum | section: faq | verification: verified | source: grand_egyptian_museum/faq.json
Q: How far is the Grand Egyptian Museum from the Pyramids of Giza?
A: The museum is located only about 2 kilometers from the Giza Plateau, and many galleries offer direct sightlines to the pyramids. Visitors commonly combine both sites in a single day.

MISSING / UNAVAILABLE INFORMATION: the query asks for exact structured data but no specific place could be identified; exact values cannot be given without a place

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```

## Query (en): Is the Giza Plateau accessible for wheelchair users?

- intent: `accessibility` - methods: `['dense', 'sparse']` - entities: `{"places": [], "city": "Giza", "structured_fields": ["accessibility_notes"]}` - latency: 32 ms
- verification_flags: `["the query asks for exact structured data but no specific place could be identified; exact values cannot be given without a place"]`

```text
⚠️ Generation is temporarily unavailable (the LLM provider could not be reached). Here is the information retrieved from the knowledge base:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### DOCUMENT giza_pyramids::accessibility::0
place_id: giza_pyramids | section: accessibility | verification: verified | source: giza_pyramids/info.json
Accessibility: Uneven sandy and rocky terrain; wheelchair access is limited on the plateau itself, though a paved viewing area is available. Pyramid interiors involve steep, low-ceilinged passages and are not wheelchair accessible.

### DOCUMENT giza_pyramids::faq_17
place_id: giza_pyramids | section: faq | verification: verified | source: giza_pyramids/faq.json
Q: Is the site wheelchair accessible?
A: Accessibility is limited. The terrain is sandy and rocky, though a paved area allows viewing from a vehicle or wheelchair at a distance. Entering pyramid interiors is not wheelchair accessible due to narrow, steep passages.

### DOCUMENT grand_egyptian_museum::faq_7
place_id: grand_egyptian_museum | section: faq | verification: verified | source: grand_egyptian_museum/faq.json
Q: Is the museum wheelchair accessible?
A: Yes, as a modern facility it includes elevators, ramps, and accessible pathways throughout its galleries and public spaces.

### DOCUMENT giza_pyramids::faq_8
place_id: giza_pyramids | section: faq | verification: verified | source: giza_pyramids/faq.json
Q: Is the Great Sphinx included in the pyramid ticket?
A: Yes, viewing the Great Sphinx is included in the general Giza Plateau entrance ticket; no separate ticket is required.

MISSING / UNAVAILABLE INFORMATION: the query asks for exact structured data but no specific place could be identified; exact values cannot be given without a place

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```

## Query (en): What is the nearest airport to the Pyramids of Giza?

- intent: `transportation` - methods: `['dense', 'sparse', 'structured', 'structured_fields']` - entities: `{"places": ["giza_pyramids"], "city": "Giza", "structured_fields": ["transportation"]}` - latency: 38 ms
- verification_flags: `[]`

```text
⚠️ Generation is temporarily unavailable (the LLM provider could not be reached). Here is the information retrieved from the knowledge base:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### STRUCTURED RECORD — Pyramids of Giza (place_id: giza_pyramids)
verification: verified (last_verified: 2026-07)
[giza_pyramids] Pyramids of Giza (أهرامات الجيزة)
City: Giza
Category: Ancient Monument / Necropolis / UNESCO World Heritage Site
Coordinates: lat 29.9792, lon 31.1342
Opening hours: 07:00–19:00 daily (Giza Plateau general entry); 08:00–17:00 daily (Giza Plateau general entry)
Ticket prices (verified): general plateau entry foreign adult: 700 EGP; great pyramid khufu interior foreign adult: 1500 EGP; great pyramid khufu interior foreign student: 750 EGP; egyptian national general entry: 150 EGP; egyptian student general entry: 75 EGP
accessibility_notes: Uneven sandy and rocky terrain; wheelchair access is limited on the plateau itself, though a paved viewing area is available. Pyramid interiors involve steep, low-ceilinged passages and are not wheelchair accessible.
nearest_airport: Cairo International Airport (CAI), approx. 30–40 km away

### DOCUMENT giza_pyramids::practical::0
place_id: giza_pyramids | section: practical | verification: verified | source: giza_pyramids/info.json
{
  "opening_hours": {
    "summer_may_to_september": "07:00–19:00 daily (Giza Plateau general entry)",
    "winter_october_to_april": "08:00–17:00 daily (Giza Plateau general entry)",
    "interior_pyramid_access": "Limited daily visitor quota for entering the Great Pyramid's interior chambers; last entry typically 1 hour before closing",
    "notes": "Hours can shift slightly for religious holidays (Ramadan, Eid) and are set by the Ministry of Tourism and Antiquities"
  },
  "ticket_price": {
    "currency": "EGP",
    "general_plateau_entry_foreign_adult": 700,
    "great_pyramid_khufu_interior_foreign_adult": 1500,
    "great_pyramid_khufu_interior_foreign_student": 750,
    "egyptian_national_general_entry": 150,
    "egyptian_student_general_entry": 75,
    "sphinx_viewing": "included in general plateau entry",
    "notes": "Prices are set by Egypt's Supreme Council of Antiquities and are revised periodically (last major revision effective January 2025). Interior access to Khafre's and Menkaure's pyramids is ticketed separately and subject to availability. Always confirm current prices via the official egymonuments.com portal or at the gate before visiting.",
    "last_verified": "2026-07"
  },
  "best_visiting_time": {
    "time_of_day": "Early morning (opening time) or 1-2 hours before closing, to avoid heat and tour bus crowds",
    "season": "October to April (cooler, more comfortable daytime temperatures)",
    "avoid": "Midday in June–August, when temperatures often exceed 38°C (100°F)"
  }
}

### DOCUMENT giza_pyramids::accessibility::1
place_id: giza_pyramids | section: accessibility | verification: verified | source: giza_pyramids/info.json
UNESCO status: Part of the UNESCO World Heritage Site 'Memphis and its Necropolis – the Pyramid Fields from Giza to Dahshur' (inscribed 1979)
Nearest airport: Cairo International Airport (CAI), approx. 30–40 km away

### DOCUMENT grand_egyptian_museum::accessibility::1
place_id: grand_egyptian_museum | section: accessibility | verification: verified | source: grand_egyptian_museum/info.json
Nearest airport: Cairo International Airport (CAI), approx. 35–45 km away

### DOCUMENT grand_egyptian_museum::faq_4
place_id: grand_egyptian_museum | section: faq | verification: verified | source: grand_egyptian_museum/faq.json
Q: How far is the Grand Egyptian Museum from the Pyramids of Giza?
A: The museum is located only about 2 kilometers from the Giza Plateau, and many galleries offer direct sightlines to the pyramids. Visitors commonly combine both sites in a single day.

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```

## Query (en): Which activities are available at Karnak Temple?

- intent: `activities` - methods: `['dense', 'sparse']` - entities: `{"places": [], "city": null, "structured_fields": []}` - latency: 40 ms
- verification_flags: `["giza_pyramids::accessibility::0"]`

```text
⚠️ Generation is temporarily unavailable (the LLM provider could not be reached). Here is the information retrieved from the knowledge base:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### DOCUMENT karnak_temple::faq_5::0
place_id: karnak_temple | section: faq | verification: verified | source: karnak_temple/faq.json
Q: Is there a Sound and Light Show at Karnak?
A: Yes, Karnak hosts an evening Sound and Light Show that narrates the temple's history while illuminating its columns, obelisks, and halls. Shows run at staggered times with rotating language narration, and tickets are purchased separately from daytime entry.

### DOCUMENT luxor_temple::faq_11
place_id: luxor_temple | section: faq | verification: verified | source: luxor_temple/faq.json
Q: Can I combine Luxor Temple and Karnak Temple in one visit?
A: Yes, both temples are on the East Bank and relatively close together; many itineraries visit Karnak in the morning and Luxor Temple in the late afternoon or evening.

### DOCUMENT karnak_temple::faq_1
place_id: karnak_temple | section: faq | verification: verified | source: karnak_temple/faq.json
Q: What are Karnak Temple's opening hours?
A: Karnak is typically open from around 6:00 am to 5:30 pm in winter (October–April) and until about 6:00 pm in summer (May–September), though exact hours can vary slightly.

### DOCUMENT giza_pyramids::accessibility::0
place_id: giza_pyramids | section: accessibility | verification: unverified | source: knowledge base
Accessibility: Uneven sandy and rocky terrain; wheelchair access is limited on the plateau itself, though a paved viewing area is available. Pyramid interiors involve steep, low-ceilinged passages and are not wheelchair accessible.

UNVERIFIED INFORMATION (present in the KB but without explicit verification): giza_pyramids::accessibility::0

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```

## Query (en): Is the Grand Egyptian Museum suitable for young children?

- intent: `age_suitability` - methods: `['dense', 'sparse', 'structured', 'structured_fields']` - entities: `{"places": ["grand_egyptian_museum"], "city": null, "structured_fields": ["age_suitability"]}` - latency: 46 ms
- verification_flags: `[]`

```text
⚠️ Generation is temporarily unavailable (the LLM provider could not be reached). Here is the information retrieved from the knowledge base:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### STRUCTURED RECORD — Grand Egyptian Museum (GEM) (place_id: grand_egyptian_museum)
verification: verified (last_verified: 2026-07)
[grand_egyptian_museum] Grand Egyptian Museum (GEM) (المتحف المصري الكبير)
City: Giza
Category: Museum
Coordinates: lat 29.9925, lon 31.1131
Opening hours: {"typical_hours": "09:00–19:00 (hours and days can vary seasonally; night hours sometimes extended for peak season)", "closed_days": "Varies by season; check official channels before visiting", "booking_requirement": "Timed, date-specific entry tickets must be booked in advance online; walk-up/on-site ticket sales are generally not available"}
Ticket prices (verified): foreign adult: 1450 EGP; foreign child student 6 to 25: 730 EGP; egyptian national adult: 200 EGP
accessibility_notes: Fully modern facility with elevators, ramps, and wheelchair accessibility throughout galleries and public areas.
nearest_airport: Cairo International Airport (CAI), approx. 35–45 km away

### DOCUMENT grand_egyptian_museum::practical::0
place_id: grand_egyptian_museum | section: practical | verification: verified | source: grand_egyptian_museum/info.json
{
  "opening_hours": {
    "typical_hours": "09:00–19:00 (hours and days can vary seasonally; night hours sometimes extended for peak season)",
    "closed_days": "Varies by season; check official channels before visiting",
    "booking_requirement": "Timed, date-specific entry tickets must be booked in advance online; walk-up/on-site ticket sales are generally not available"
  },
  "ticket_price": {
    "currency": "EGP",
    "foreign_adult": 1450,
    "foreign_child_student_6_to_25": 730,
    "egyptian_national_adult": 200,
    "children_under_6": "free",
    "notes": "Prices rose significantly after the November 2025 full opening compared to earlier soft-opening rates (some older sources cite EGP 500-600, which are outdated). Tickets are sold only through the official portal (visit-gem.com) and are non-refundable and non-transferable once booked. Beware of unofficial or fraudulent ticketing websites.",
    "last_verified": "2026-07"
  },
  "best_visiting_time": {
    "time_of_day": "Morning slots are less crowded; late-afternoon slots allow combining with sunset views of the pyramids from museum terraces",
    "season": "Year-round (indoor, climate-controlled), though cooler months (October–April) pair better with an outdoor pyramid visit the same day",
    "duration_recommended": "3 to 6 hours depending on depth of interest"
  }
}

### DOCUMENT giza_pyramids::faq_15
place_id: giza_pyramids | section: faq | verification: verified | source: giza_pyramids/faq.json
Q: Can children enter the pyramids for free?
A: Young children (typically under 6) are usually granted free entry at most Egyptian archaeological sites, though policies can vary; it's best to check current rules at the ticket office.

### DOCUMENT grand_egyptian_museum::faq_3::1
place_id: grand_egyptian_museum | section: faq | verification: verified | source: grand_egyptian_museum/info.json
Q: Is the Grand Egyptian Museum the same as the old Egyptian Museum in Tahrir Square?
A: . The Grand Egyptian Museum is a newly built, much larger facility near the Giza Pyramids that has taken over most major collections, including the full Tutankhamun collection, though the Tahrir museum remains open with a reduced display.

### DOCUMENT grand_egyptian_museum::faq_1
place_id: grand_egyptian_museum | section: faq | verification: verified | source: grand_egyptian_museum/info.json
Q: How much does it cost to enter the Grand Egyptian Museum?
A: As of the full opening, foreign adult tickets cost 1,450 EGP (around $30 USD), while foreign children and students aged 6–25 pay 730 EGP. Egyptian nationals pay a much lower rate of around 200 EGP. Children under 6 enter free.

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```

## Query (en): What are the best places to visit in Aswan?

- intent: `recommendation` - methods: `['dense', 'sparse']` - entities: `{"places": [], "city": null, "structured_fields": []}` - latency: 37 ms
- verification_flags: `["karnak_temple::faq_8", "giza_pyramids::faq_7"]`

```text
⚠️ Generation is temporarily unavailable (the LLM provider could not be reached). Here is the information retrieved from the knowledge base:


RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):

### DOCUMENT abu_simbel::practical::0
place_id: abu_simbel | section: practical | verification: verified | source: abu_simbel/info.json
{
  "opening_hours": {
    "winter_october_to_april": "06:00–17:00 daily",
    "summer_may_to_september": "05:00–18:00 daily",
    "sun_festival_days": "Special early access typically offered on February 22 and October 22 for the Sun Festival"
  },
  "ticket_price": {
    "currency": "EGP",
    "foreign_adult_standard_day": 750,
    "foreign_student_standard_day": 375,
    "sun_festival_day_foreign_adult": 1200,
    "sun_festival_day_foreign_student": 600,
    "notes": "Prices vary somewhat between sources and are set by the Ministry of Tourism and Antiquities; figures cited here reflect commonly reported 2026 rates (roughly USD equivalents of $15–30). Sun Festival dates carry premium pricing due to high demand. Confirm current prices via the official egymonuments.com portal before traveling.",
    "last_verified": "2026-07"
  },
  "best_visiting_time": {
    "time_of_day": "Early morning, especially for the Sun Festival sunrise alignment (dawn)",
    "season": "October to April for milder daytime temperatures; the Sun Festival occurs on February 22 and October 22 each year",
    "travel_note": "Most visitors arrive via a roughly 45-minute flight from Aswan or a 3.5–4 hour road trip; flights and roads can occasionally be affected by desert sandstorms"
  }
}

### DOCUMENT karnak_temple::faq_8
place_id: karnak_temple | section: faq | verification: unverified | source: knowledge base
Q: What is the best time of day to visit Karnak?
A: Early morning right at opening offers cooler temperatures and fewer tour groups; late afternoon after large groups depart is another good window. Midday hours combine peak heat and crowds and are best avoided.

### DOCUMENT abu_simbel::overview::4
place_id: abu_simbel | section: overview | verification: verified | source: abu_simbel/info.json
. In one of the 20th century's most celebrated feats of engineering, both temples were cut into large blocks and relocated in the 1960s to higher ground to save them from being submerged by the rising waters of Lake Nasser following the construction of the Aswan High Dam, in a UNESCO-led

### DOCUMENT giza_pyramids::faq_7
place_id: giza_pyramids | section: faq | verification: unverified | source: knowledge base
Q: What is the best time of day to visit the pyramids?
A: Early morning right at opening, or the last one to two hours before closing, offer cooler temperatures, softer light for photography, and smaller crowds compared to midday.

UNVERIFIED INFORMATION (present in the KB but without explicit verification): karnak_temple::faq_8; giza_pyramids::faq_7

RULES: use ONLY the evidence above. Do not invent prices, hours, coordinates or transport routes. Mark unverified/estimated values as such. Cite evidence as [doc_id or place_id].
```
