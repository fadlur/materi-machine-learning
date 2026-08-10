"""
=============================================================
FASE 8 - MODUL 3: RESUME DAN PORTFOLIO GUIDE
=============================================================
Resume dan portfolio adalah GATEWAY ke interview.
Tanpa yang bagus, skills-mu tidak akan terlihat.

Goal:
- Resume: 30-second scan -> "This person looks promising"
- Portfolio: 5-minute review -> "This person can deliver"

Koneksi Teknik Elektro:
- Resume = datasheet dari capabilities-mu
- Portfolio = demo reel dari projects-mu
- LinkedIn = broadcast channel untuk skills-mu
- GitHub = source code repository (open for inspection)

Durasi target: 2-3 jam (dengan iterative improvement)
============================================================="""

import numpy as np

np.random.seed(42)


# ===========================================================
# BAGIAN 1: Resume Structure untuk ML Engineer
# ===========================================================
print("="*60)
print("BAGIAN 1: STRUKTUR RESUME ML ENGINEER")
print("="*60)

resume_structure = """
TARGET FORMAT:
- 1 page (untuk <5 years experience)
- 2 pages (untuk 5+ years experience)
- ATS-friendly (bisa di-parse oleh applicant tracking systems)
- PDF format

ATS (Applicant Tracking System) adalah software yang digunakan
recruiters untuk screen resume. Tips ATS-friendly:
- Gunakan standard section headings (Experience, Education, Skills)
- Avoid tables, columns, images, dan fancy formatting
- Gunakan keywords dari job description
- Save sebagai PDF (bukan Word atau image)

PANDUAN SECTIONS (dalam urutan):

1. HEADER
   - Nama, lokasi, email, phone
   - LinkedIn URL, GitHub URL, portfolio URL
   - Opsional: personal website
   
   TIPS:
   - Gunakan professional email (bukan username aneh)
   - LinkedIn dan GitHub adalah MUST untuk ML Engineer
   - Personal website bisa differentiate kamu dari kandidat lain

2. SUMMARY (2-3 kalimat)
   - Role yang di-target
   - Years of experience
   - Key strengths (2-3 areas)
   - Impact yang di-deliver
   
   Example:
   "ML Engineer dengan 3 tahun experience di deployment
   production ML systems. Spesialisasi di computer vision
   dan MLOps. Delivered 5+ models ke production dengan
   99.9% uptime dan $500K annual savings."
   
   TIPS:
   - Summary harus spesifik dan quantified
   - Hindari buzzwords tanpa bukti ("passionate", "hardworking")
   - Tailor summary per job application

3. TECHNICAL SKILLS
   - Group by category
   - Programming: Python, SQL, C++
   - ML/DL: PyTorch, TensorFlow, scikit-learn, XGBoost
   - MLOps: Docker, Kubernetes, AWS SageMaker, MLflow
   - Data: Spark, Pandas, NumPy, SQL
   - Tools: Git, Linux, Jupyter, VS Code
   
   TIPS: jangan list semua, fokus pada yang relevant
   HINDARI: "Microsoft Office", "HTML" (untuk ML role)
   
   DETAIL:
   - Pilih 8-15 skills yang paling relevant
   - Urutkan dari yang paling kuat/paling relevant
   - Jangan list version numbers (akan outdated)
   - Hindari rating diri sendiri (e.g., "Python: 5/5")

4. EXPERIENCE (reverse chronological)
   Format:
   - Company | Role | Dates
   - Bullet points: accomplishment-oriented
   
   STRUKTUR BULLET POINT:
   [Action Verb] + [What you did] + [How you did it] + [Impact]
   
   X Bad: "Built machine learning models"
   OK Good: "Built and deployed CNN-based defect detection
            model yang mengurangi false positives by 40%
            dan save $200K annually"
   
   PANDUAN METRICS YANG BAGUS:
   - Performance: accuracy, precision, recall, F1, AUC
   - Scale: QPS, latency, throughput
   - Business: revenue, cost savings, efficiency gain
   - Time: speedup, reduction in processing time
   
   TIPS: Gunakan numbers whenever possible!
   
   DETAIL:
   - Action verbs: Built, Deployed, Optimized, Reduced, Improved,
     Led, Designed, Implemented, Automated, Scaled.
   - Quantify impact: "reduced latency by 50%" lebih kuat dari
     "improved latency".
   - Context: seberapa besar masalahnya? seberapa complex solusinya?

5. PROJECTS (jika experience sedikit)
   - Nama project
   - Tech stack
   - 2-3 bullet points dengan metrics
   - Link ke GitHub/demo
   
   Example:
   "Predictive Maintenance System (Python, PyTorch, FastAPI)
    - Built LSTM model untuk predict equipment failure
      dengan 92% precision dan 48-hour lead time
    - Deployed sebagai REST API dengan Docker,
      handling 1000 requests/minute
    - Reduced unplanned downtime by 35%"
   
   TIPS:
   - Projects harus end-to-end (bukan hanya model training)
   - Mention business impact, bukan hanya accuracy
   - Link harus clickable dan mengarah ke repo yang rapi

6. EDUCATION
   - Degree, Institution, Year
   - Relevant coursework (jika baru graduate)
   - GPA (opsional, hanya jika >3.5)
   
   TIPS:
   - Untuk career changer: highlight relevant coursework
   - Certifications: Coursera, AWS, GCP (jika relevant)
   - Untuk S2: sebutkan thesis/research topic jika relevant

7. CERTIFICATIONS (opsional)
   - AWS Certified Machine Learning
   - Google Cloud Professional ML Engineer
   - TensorFlow Developer Certificate
   - etc.
   
   TIPS:
   - Certifications cloud (AWS/GCP/Azure) sangat valuable
   - Deep Learning Specialization (Coursera) dihargai
   - Jangan list certifications yang tidak relevant

TARGET KEYWORDS UNTUK ATS:
- Machine Learning, Deep Learning, Neural Networks
- Python, PyTorch, TensorFlow, scikit-learn
- MLOps, Docker, Kubernetes, CI/CD
- SQL, Spark, BigQuery, Data Pipeline
- AWS, GCP, Azure
- Computer Vision, NLP, Time Series

DETAIL:
- Baca job description dan extract keywords
- Pastikan keywords tersebut muncul di resume
- Gunakan Jobscan.co untuk ATS optimization
"""
print(resume_structure)


# ===========================================================
# BAGIAN 2: Portfolio Projects
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 2: PORTFOLIO PROJECTS")
print("="*60)

portfolio_guide = """
TARGET PRINSIP PORTFOLIO:

Quality > Quantity
- 3 projects yang bagus > 10 projects yang mediocre
- Setiap project harus demonstrate different skills
- End-to-end > partial implementation

JENIS PROJECTS YANG BAGUS:

1. END-TO-END ML PIPELINE
   - Data collection -> Feature engineering -> Model -> Deployment
   - Demonstrate: full ML lifecycle
   - Example: Predictive maintenance system
   
   DETAIL:
   - Harus ada data ingestion (bukan hanya load CSV)
   - Feature engineering yang thoughtful
   - Model selection dengan justification
   - Evaluation yang comprehensive
   - Deployment sebagai API atau app

2. PRODUCTION-READY DEPLOYMENT
   - API, Docker, monitoring, CI/CD
   - Demonstrate: MLOps skills
   - Example: Image classification API dengan auto-scaling
   
   DETAIL:
   - FastAPI/Flask dengan proper error handling
   - Dockerfile dengan multi-stage build
   - Docker Compose untuk local development
   - GitHub Actions untuk CI/CD
   - Logging dan monitoring dasar

3. RESEARCH IMPLEMENTATION
   - Implement paper dari scratch
   - Demonstrate: deep understanding
   - Example: Transformer dari scratch
   
   DETAIL:
   - Pilih paper yang well-known (Attention Is All You Need, ResNet)
   - Implement dari scratch dengan NumPy/PyTorch
   - Bandingkan dengan library implementation
   - Dokumentasi yang menjelaskan setiap component

4. DOMAIN-SPECIFIC APPLICATION
   - Apply ML ke domain yang kamu kenal
   - Demonstrate: domain knowledge + ML
   - Example: Power quality classification untuk EE background
   
   DETAIL:
   - Gunakan domain knowledge untuk feature engineering
   - Explain kenapa ML cocok untuk problem ini
   - Bandingkan dengan traditional methods
   - Show business value

TARGET STRUKTUR README UNTUK SETIAP PROJECT:

```markdown
# Project Name

## Overview
1-2 paragraphs: what, why, how

## Demo
[Link ke demo video atau live app]

## Architecture
[Diagram atau explanation]

## Features
- Feature 1: description
- Feature 2: description

## Tech Stack
- Framework: PyTorch/TensorFlow
- Deployment: Docker/AWS/GCP
- Monitoring: MLflow/Evidently

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 95% |
| Latency | 50ms |

## Installation
```bash
git clone ...
cd project
pip install -r requirements.txt
```

## Usage
```bash
python train.py
python deploy.py
```

## Lessons Learned
What worked, what didn't, what you'd do differently
```

DETAIL:
- README adalah dokumen paling penting di setiap project
- Recruiter akan baca README sebelum melihat code
- Sertakan screenshot atau demo video
- "Lessons Learned" menunjukkan growth mindset

TARGET GITHUB PROFILE OPTIMIZATION:

1. Pin 3-6 best repositories
2. Set profile README (github.com/username/username)
3. Consistent naming convention
4. Clean code with comments
5. Unit tests
6. CI/CD badges
7. Contribution graph (consistency!)

TIPS Profile README template:
```markdown
## Hi, I'm [Name]

ML Engineer dengan background Teknik Elektro.
Passionate tentang bridging theory dan practice di AI.

### Current Focus
- Deploying production ML systems
- Computer vision untuk industrial inspection
- MLOps best practices

### Tech Stack
Python | PyTorch | Docker | AWS | Kubernetes

### Connect
[LinkedIn] [Email] [Website]
```

DETAIL:
- GitHub profile README adalah "landing page" gratis
- Contribution graph yang konsisten menunjukkan dedication
- Pin repositories yang paling impressive
- Setiap repo harus punya README yang rapi
"""
print(portfolio_guide)


# ===========================================================
# BAGIAN 3: LinkedIn Optimization
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 3: LINKEDIN OPTIMIZATION")
print("="*60)

linkedin_guide = """
TARGET HEADLINE:
Format: [Role] | [Specialization] | [Value Proposition]

Examples:
- "ML Engineer | Computer Vision & MLOps | Building production AI systems"
- "Data Scientist | NLP & Transformers | Turning data into decisions"
- "AI Engineer | LLMs & RAG | Deploying intelligent applications"

X Bad: "Looking for opportunities"
X Bad: "Student at XYZ University"
OK Good: Specific dan value-oriented

DETAIL:
- Headline adalah hal pertama yang dilihat recruiter
- Gunakan keywords yang recruiter cari
- Jangan terlalu generic
- Update headline sesuai target role

TARGET ABOUT SECTION:

Structure:
1. Hook (1-2 kalimat yang menarik)
2. What you do (2-3 kalimat)
3. Key achievements (dengan metrics)
4. What you're looking for

Example:
"I build machine learning systems that solve real problems.

Dengan background Teknik Elektro dan 3 tahun experience di ML,
saya specialize di production deployment dari computer vision
models. Recent projects include:
- Defect detection system dengan 95% accuracy, deployed di
  5 manufacturing plants
- Predictive maintenance model yang mengurangi downtime 30%
- End-to-end MLOps pipeline dengan CI/CD dan monitoring

Passionate tentang bridging theory dan practice - dari research
paper ke production system.

Currently open untuk ML Engineer roles di tech companies
atau AI startups."

DETAIL:
- About section adalah tempat untuk storytelling
- Gunakan bullet points untuk readability
- Sertakan metrics dan achievements
- Call-to-action di akhir (what you're looking for)

TARGET EXPERIENCE SECTIONS:
- Sama dengan resume, tapi bisa lebih detailed
- Gunakan rich media (images, links, documents)
- Tag companies dan colleagues
- Request recommendations dari managers/peers

DETAIL:
- Recommendations dari senior colleagues sangat powerful
- Rich media (presentations, demo videos) menarik perhatian
- Tag orang yang bekerja sama di project

TARGET SKILLS & ENDORSEMENTS:
- List 20-50 relevant skills
- Pin top 3 yang paling relevant
- Get endorsements dari connections
- Take LinkedIn Skill Assessments

DETAIL:
- Skills yang di-endorse oleh banyak orang lebih credible
- LinkedIn Skill Assessments menunjukkan competency
- Jangan tambahkan skills yang tidak dikuasai

TARGET ACTIVITY:
- Post consistently (1-2x per week)
- Share projects, learnings, insights
- Comment dan engage dengan posts orang lain
- Write articles (showcase expertise)

TIPS CONTENT IDEAS:
- "How I built [project] - lessons learned"
- "Understanding [concept] dengan analogi sederhana"
- "Comparison: [tool A] vs [tool B]"
- "Tutorial: [specific technique]"
- "Behind the scenes: [project]"

DETAIL:
- Consistency > viral content
- Posting regularly membuat kamu visible di feed recruiter
- Comment di posts thought leaders = networking pasif
- Articles menunjukkan deep expertise

TARGET NETWORKING:
- Connect dengan recruiters di target companies
- Join ML/AI groups
- Attend virtual events dan webinars
- Follow thought leaders
- Engage dengan their content

DETAIL:
- Personalized connection request lebih baik dari default
- Jangan hanya connect, tapi juga engage
- Informational interviews = cara terbaik untuk network
- Follow up setelah bertemu di event
"""
print(linkedin_guide)


# ===========================================================
# BAGIAN 4: Interview Preparation
# ===========================================================
print("\n" + "="*60)
print("BAGIAN 4: INTERVIEW PREPARATION")
print("="*60)

interview_prep = """
TARGET BEFORE INTERVIEW:

1. RESEARCH COMPANY
   - Products/services
   - Tech stack
   - Recent news
   - Culture dan values
   - Interview process
   
   DETAIL:
   - Baca blog engineering company untuk understand tech stack
   - Glassdoor untuk interview experiences
   - Recent news untuk talking points
   - Company values untuk behavioral questions

2. PREPARE STORIES
   Gunakan STAR method:
   - Situation: context dari story
   - Task: apa yang perlu di-achieve
   - Action: apa yang kamu lakukan
   - Result: outcome (dengan metrics!)
   
   Contoh stories:
   - Challenge yang di-overcome
   - Mistake yang di-learn dari
   - Conflict yang di-resolve
   - Project yang di-lead
   - Innovation yang di-introduce
   
   DETAIL:
   - Siapkan 5-8 stories yang versatile
   - Setiap story harus punya metrics
   - Practice telling them dalam 2-3 menit
   - Record yourself untuk cek clarity

3. PREPARE QUESTIONS
   Tanyakan ke interviewer:
   - "What does success look like in this role?"
   - "What are the biggest challenges the team faces?"
   - "How does the team approach MLOps?"
   - "What's the typical project lifecycle?"
   - "How does the company support learning?"
   
   DETAIL:
   - Pertanyaan yang bagus menunjukkan interest dan research
   - Hindari pertanyaan yang jawabannya ada di website
   - Tanyakan tentang growth dan development
   - Tanyakan tentang day-to-day work

TARGET DURING INTERVIEW:

1. BEHAVIORAL QUESTIONS (Amazon LP style)
   "Tell me about a time when..."
   - You had to learn something quickly
   - You disagreed with a teammate
   - You failed at something
   - You went above and beyond
   - You had to prioritize competing tasks
   
   TIPS: always use STAR method
   TIPS: Have 5-8 stories ready
   
   DETAIL:
   - Behavioral questions menilai culture fit
   - Jangan bohong - recruiter bisa detect inconsistency
   - Fokus pada "I" bukan "we" (what YOU did)
   - Selalu sebutkan lessons learned

2. TECHNICAL QUESTIONS
   - Think out loud
   - Clarify requirements
   - Discuss tradeoffs
   - Test dengan examples
   - Optimize setelah brute force
   
   DETAIL:
   - Thinking out loud membantu interviewer understand your process
   - Clarify requirements sebelum coding
   - Discuss tradeoffs menunjukkan depth of knowledge
   - Selalu test dengan contoh input

3. SYSTEM DESIGN
   - Start dengan requirements
   - Draw diagram
   - Quantify (numbers!)
   - Discuss tradeoffs
   - Mention monitoring
   
   DETAIL:
   - System design interview menilai engineering judgment
   - Jangan langsung jump ke solution
   - Quantify: "fast" -> "p99 < 100ms"
   - Mention operational aspects (monitoring, rollback)

TARGET AFTER INTERVIEW:

1. THANK YOU EMAIL (24 jam)
   - Thank interviewer for their time
   - Reiterate interest
   - Mention specific discussion points
   - Keep it brief (3-4 sentences)
   
   DETAIL:
   - Thank you email membuat kamu memorable
   - Mention specific topics yang didiskusikan
   - Reiterate fit untuk role
   - Keep it professional dan concise

2. FOLLOW UP
   - Jika belum ada kabar dalam timeline yang dijanjikan
   - Polite follow-up email
   - Reiterate interest
   
   DETAIL:
   - Follow up setelah 1 week jika tidak ada response
   - Jangan terlalu aggressive (maksimal 2 follow-ups)
   - Keep it brief dan professional

3. REFLECT
   - Apa yang berjalan baik?
   - Apa yang bisa diperbaiki?
   - Questions yang sulit?
   - Update preparation materials
   
   DETAIL:
   - Refleksi adalah kunci untuk improvement
   - Catat questions yang sulit untuk dipelajari nanti
   - Update cheat sheet jika ada konsep baru

TARGET SALARY NEGOTIATION:

1. RESEARCH
   - Levels.fyi (salary data)
   - Glassdoor
   - Blind (anonymous discussions)
   - Peer discussions
   
   DETAIL:
   - Research salary range untuk role, level, dan lokasi
   - Total compensation = base + equity + bonus + benefits
   - Equity bisa lebih valuable daripada base untuk startups

2. KNOW YOUR WORTH
   - Years of experience
   - Skills yang rare
   - Track record
   - Location
   
   DETAIL:
   - ML Engineer dengan MLOps skills bisa negotiate lebih tinggi
   - Background backend/DevOps adalah keunggulan
   - Portfolio projects bisa justify higher salary

3. NEGOTIATION TIPS
   - Jangan disclose current salary
   - Let them make first offer
   - Negotiate total compensation (base + equity + bonus)
   - Consider non-monetary (remote, learning budget, etc.)
   - Always negotiate (worst case: they say no)
   - Get offer in writing
   
   DETAIL:
   - Non-monetary benefits bisa valuable:
     * Remote work flexibility
     * Learning budget (conferences, courses)
     * Stock options / RSUs
     * Sign-on bonus
     * Relocation assistance

TIPS NEGOTIATION SCRIPT:
"Thank you untuk offer-nya. Saya excited tentang opportunity ini.
Berdasarkan research dan experience saya, saya expecting
range [X-Y]. Apakah ada flexibility di sana?"

TARGET ACCEPTING/DECLINING:

Accepting:
- Get written offer
- Understand equity (vesting schedule, cliff)
- Negotiate start date
- Give notice (2 weeks standard)
- Maintain relationships

Declining:
- Be polite dan professional
- Give brief reason
- Keep door open
- Thank them untuk opportunity
"""
print(interview_prep)


# ===========================================================
# LATIHAN 21: Build Your Portfolio
# ===========================================================
"""
TARGET Learning Objectives:
   - Membuat resume yang ATS-friendly dan compelling
   - Membangun portfolio dengan 3+ projects
   - Optimizing LinkedIn profile
   - Preparing untuk behavioral interviews

PANDUAN LANGKAH-LANGKAH:

STEP 1: Resume Draft
--------------------
   a) Gunakan template dari section 1
   b) Isi dengan experience dan projects
   c) Quantify semua accomplishments
   d) Gunakan action verbs:
      - Built, Deployed, Optimized, Reduced, Improved
      - Led, Designed, Implemented, Automated, Scaled
   
   e) Review dengan checklist:
      [ ] 1-2 pages
      [ ] ATS-friendly (no tables, no images)
      [ ] Keywords untuk target roles
      [ ] Metrics di setiap bullet point
      [ ] No typos atau grammar errors
      [ ] Consistent formatting
      
   f) Get feedback dari 2-3 people
   g) Iterate 3-5x
   
   DETAIL:
   - Setiap iterasi harus lebih baik dari sebelumnya
   - Feedback dari senior ML Engineer sangat valuable
   - Gunakan Grammarly untuk grammar check
   - Print untuk review (berbeda feel dari screen)


STEP 2: Portfolio Projects
--------------------------
   a) Pilih 3 projects yang demonstrate different skills:
      - Project 1: End-to-end ML pipeline
      - Project 2: Production deployment/MLOps
      - Project 3: Domain-specific application
      
   b) Untuk setiap project:
      [ ] Clean code dengan comments
      [ ] Comprehensive README
      [ ] Requirements.txt
      [ ] Demo atau screenshots
      [ ] Results dengan metrics
      
   c) Publish ke GitHub
   d) Pin repositories
   e) Create GitHub profile README
   
   DETAIL:
   - README adalah dokumen paling penting
   - Code harus bisa di-run tanpa error
   - Sertakan requirements.txt dengan version pinning
   - Demo video bisa lebih impactful daripada screenshots


STEP 3: LinkedIn Optimization
-----------------------------
   a) Update headline
   b) Write compelling About section
   c) Add experience dengan rich media
   d) List relevant skills
   e) Request 2-3 recommendations
   f) Start posting (1-2x per week)
   g) Connect dengan 50+ people di industry
   
   DETAIL:
   - Headline harus mengandung keywords target role
   - About section = mini cover letter
   - Recommendations dari managers > peers
   - Consistent posting = visibility


STEP 4: Interview Preparation
-----------------------------
   a) Prepare 5-8 STAR stories
   b) Practice behavioral questions
   c) Do 3+ mock interviews
   d) Research 10 target companies
   e) Prepare questions untuk ask interviewer
   
   DETAIL:
   - Mock interview dengan peer atau mentor
   - Record dan review responses
   - Research company culture dan values
   - Siapkan questions yang thoughtful


STEP 5: Job Application Strategy
--------------------------------
   a) Target companies:
      - Dream companies (5)
      - Good fit (10)
      - Safety (5)
      
   b) Application channels:
      - Direct application (company website)
      - LinkedIn Easy Apply
      - Referrals (paling efektif!)
      - Recruiters
      
   c) Follow up:
      - Track applications (spreadsheet)
      - Follow up setelah 1-2 weeks
      - Leverage network untuk referrals
      
   d) Interview loop:
      - Phone screen
      - Technical (coding/ML)
      - System design
      - Behavioral
      - Onsite/Virtual onsite
      - Offer


TIPS:
   - Tailor resume per job posting
   - Use Jobscan.co untuk ATS optimization
   - Network > cold application
   - Practice behavioral dengan STAR method
   - Follow up politely
   - Track semua applications

PERINGATAN COMMON MISTAKES:
   - Generic resume untuk semua applications
   - Tidak quantify accomplishments
   - Portfolio projects tidak end-to-end
   - LinkedIn yang outdated
   - Tidak prepare untuk behavioral
   - Apply tanpa research company
   - Tidak negotiate offer

TARGET EXPECTED OUTPUT:
   - Polished resume (1-2 pages)
   - GitHub portfolio dengan 3+ projects
   - Optimized LinkedIn profile
   - 5-8 STAR stories
   - Application tracking system
   - Job search strategy

Your career is in your hands - make it count!
"""


# ===========================================================
# 🔥 CHALLENGE: 90-Day Job Search Sprint
# ===========================================================
"""
TARGET Learning Objectives:
   - Execute structured job search dalam 90 hari
   - Mengubah portfolio menjadi job offers
   - Membangun network di industry

PANDUAN LANGKAH-LANGKAH:

WEEK 1-2: FOUNDATION
--------------------
[ ] Finalize resume (3-5 iterations)
[ ] Complete 3 portfolio projects
[ ] Optimize LinkedIn profile
[ ] Set up GitHub profile README
[ ] Create personal website (opsional)
[ ] Draft cover letter template
[ ] Research 20 target companies

DETAIL:
- Foundation adalah kunci untuk everything else
- Jangan rush - quality matters
- Personal website bisa differentiate kamu

WEEK 3-4: NETWORKING
--------------------
[ ] Connect dengan 50+ people di target companies
[ ] Join 3+ ML/AI communities (Slack, Discord, LinkedIn)
[ ] Attend 2+ virtual events atau meetups
[ ] Reach out untuk informational interviews (5-10)
[ ] Ask for referrals dari connections
[ ] Engage dengan content di LinkedIn (daily)

DETAIL:
- Networking adalah kunci untuk referrals
- Informational interviews = networking tanpa pressure
- Communities = tempat belajar dan networking

WEEK 5-8: APPLICATIONS
----------------------
[ ] Apply ke 5-10 jobs per week
[ ] Tailor resume per application
[ ] Follow up setelah 1-2 weeks
[ ] Track semua applications
[ ] Prepare untuk interviews (coding, system design, behavioral)
[ ] Do mock interviews (weekly)

DETAIL:
- Quality > quantity (tailor resume per application)
- Track applications di spreadsheet
- Follow up adalah kunci
- Mock interviews harus regular

WEEK 9-10: INTERVIEWS
---------------------
[ ] Phone screens
[ ] Technical rounds
[ ] System design rounds
[ ] Behavioral rounds
[ ] Take-home assignments (jika ada)
[ ] Debrief setelah setiap interview

DETAIL:
- Setiap interview adalah learning opportunity
- Debrief untuk identify improvement areas
- Jangan discouraged oleh rejection

WEEK 11-12: OFFERS & DECISION
-----------------------------
[ ] Evaluate offers (compensation, role, team, growth)
[ ] Negotiate (base, equity, bonus, benefits)
[ ] Make decision
[ ] Accept offer
[ ] Give notice (jika currently employed)
[ ] Celebrate!

PANDUAN METRICS TO TRACK:
- Applications sent per week
- Response rate
- Interview conversion rate
- Time dari application ke offer
- Network growth (new connections per week)

TIPS:
- Consistency > intensity
- Quality applications > quantity
- Follow up adalah kunci
- Network adalah everything
- Jangan take rejection personally
- Setiap interview adalah learning opportunity

PERINGATAN COMMON MISTAKES:
- Apply tanpa tailoring resume
- Ignore networking
- Underprepare untuk interviews
- Take first offer tanpa negotiate
- Burn out dari terlalu intense
- Give up too early (average job search: 3-6 months)

TARGET SUCCESS INDICATORS:
- 5+ interviews dalam 90 hari
- 2+ offers
- Strong network (100+ relevant connections)
- Confidence dalam technical interviews

You got this!
"""

print("\n" + "="*50)
print("SELESAI FASE 8!")
print("="*50)
print("""
Kamu sekarang siap untuk:
OK Technical interviews (coding, ML theory)
OK System design interviews
OK Behavioral interviews
OK Resume dan portfolio
OK Networking dan job search

Sebelum lanjut:
1. Finalize resume dan portfolio
2. Practice mock interviews
3. Start job applications

Lanjut ke: Fase 9 - Apply & Iterate (lihat README.md)
""")
