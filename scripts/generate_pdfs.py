#!/usr/bin/env python3
"""
Generate PDF sample CVs for data/raw/.

Run ONCE locally before starting Docker:
    pip install reportlab arabic-reshaper python-bidi
    python scripts/generate_pdfs.py

Produces 6 PDF files in data/raw/ (5 English + 1 Arabic).
After running, the HTML files in data/raw/ can be deleted if you want PDF-only ingestion.
"""
import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("Error: reportlab not installed.")
    print("Run:  pip install reportlab")
    sys.exit(1)

styles = getSampleStyleSheet()
H1    = ParagraphStyle('H1',    parent=styles['Heading1'], fontSize=18, spaceAfter=6)
H2    = ParagraphStyle('H2',    parent=styles['Heading2'], fontSize=12, spaceAfter=4, spaceBefore=10)
BODY  = ParagraphStyle('Body',  parent=styles['Normal'],   fontSize=10, spaceAfter=4, leading=14)
BULLET= ParagraphStyle('Bul',   parent=styles['Normal'],   fontSize=10, leftIndent=16, spaceAfter=3, leading=14)


def _doc(filename):
    return SimpleDocTemplate(
        os.path.join(OUTPUT_DIR, filename),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

def hr():    return HRFlowable(width="100%", thickness=0.5, color='grey', spaceAfter=6, spaceBefore=2)
def h1(t):  return Paragraph(t, H1)
def h2(t):  return Paragraph(t, H2)
def p(t):   return Paragraph(t, BODY)
def li(t):  return Paragraph(f"• {t}", BULLET)


# ─── CV 1: John Smith ────────────────────────────────────────────────────────
def cv_john_smith():
    doc = _doc('cv_john_smith.pdf')
    doc.build([
        h1("John Smith"),
        p("john.smith@email.com | +1-415-555-0101 | San Francisco, CA | github.com/jsmith-dev"),
        hr(),
        h2("Summary"),
        p("Senior Backend Engineer with 7 years of experience building high-throughput Python services "
          "and REST APIs. Deep expertise in FastAPI, PostgreSQL, and distributed systems. Led backend "
          "teams at two YC-backed startups. Passionate about clean architecture and developer tooling."),
        hr(),
        h2("Technical Skills"),
        li("Languages: Python (expert), Go (proficient), SQL, Bash"),
        li("Frameworks: FastAPI, Django, SQLAlchemy, Celery, pytest"),
        li("Databases: PostgreSQL, Redis, MongoDB, Elasticsearch"),
        li("DevOps: Docker, Kubernetes, GitHub Actions, AWS (EC2, RDS, S3, Lambda)"),
        li("ML/AI: PyTorch (basic), Hugging Face Transformers, LangChain, RAG pipelines"),
        hr(),
        h2("Work Experience"),
        h2("Senior Backend Engineer — DataStream Inc., San Francisco (Jan 2021 – Present)"),
        li("Architected FastAPI microservices handling 50,000 req/s with <30 ms p99 latency using async Python + Postgres."),
        li("Designed real-time event pipeline using Kafka + Celery processing 2 million events/day."),
        li("Reduced API response times 40% through query optimisation and Redis caching layer."),
        li("Mentored 4 junior engineers; led bi-weekly architecture review sessions."),
        h2("Backend Engineer — HireAI (YC W19), San Francisco (Mar 2018 – Dec 2020)"),
        li("Built resume parsing pipeline (FastAPI + SpaCy) processing 10,000 CVs/day."),
        li("Implemented full-text search with Elasticsearch; improved candidate match recall by 25%."),
        li("Designed PostgreSQL schema for multi-tenant SaaS with row-level security (RLS)."),
        hr(),
        h2("Education"),
        p("B.Sc. Computer Science — UC Berkeley (2017) | GPA: 3.8/4.0"),
        hr(),
        h2("Certifications"),
        li("AWS Certified Solutions Architect — Associate, 2022"),
        li("Certified Kubernetes Application Developer (CKAD) — CNCF, 2021"),
    ])
    print("OK cv_john_smith.pdf")


# ─── CV 2: Sarah Chen ─────────────────────────────────────────────────────────
def cv_sarah_chen():
    doc = _doc('cv_sarah_chen.pdf')
    doc.build([
        h1("Sarah Chen"),
        p("sarah.chen@email.com | +1-206-555-0202 | Seattle, WA | linkedin.com/in/sarahchen-ml"),
        hr(),
        h2("Summary"),
        p("Machine Learning Engineer and NLP Researcher with 5 years building production ML systems. "
          "Specialised in large language models, RAG pipelines, and embedding-based semantic retrieval. "
          "Published 2 papers at ACL. PhD in Computational Linguistics from University of Washington."),
        hr(),
        h2("Technical Skills"),
        li("ML Frameworks: PyTorch, TensorFlow, Hugging Face Transformers, sentence-transformers"),
        li("NLP: BERT, GPT, RAG pipelines, Named Entity Recognition, semantic search, embeddings"),
        li("Vector Databases: ChromaDB, Pinecone, FAISS, Weaviate"),
        li("Languages: Python (expert), R, SQL, Bash"),
        li("MLOps: MLflow, DVC, Weights & Biases, Docker, Kubernetes"),
        li("Cloud: AWS SageMaker, GCP Vertex AI, Azure ML"),
        hr(),
        h2("Work Experience"),
        h2("Senior ML Engineer — Amazon Alexa AI, Seattle (Jun 2021 – Present)"),
        li("Built multilingual intent classification system (15 languages) deployed to 300 million Alexa devices."),
        li("Designed RAG pipeline for Alexa Q&A using sentence-transformers + DynamoDB; improved accuracy by 18%."),
        li("Reduced model inference latency 3× through ONNX quantisation and TensorRT optimisation."),
        h2("ML Engineer — Semantic Health, Seattle (Sep 2018 – May 2021)"),
        li("Developed clinical NLP pipeline extracting 40+ entity types from medical notes using BioBERT."),
        li("Fine-tuned GPT-2 for clinical text summarisation; saved 3 hours per physician per week."),
        hr(),
        h2("Education"),
        p("Ph.D. Computational Linguistics — University of Washington (2019) | GPA: 3.95/4.0"),
        p("B.Sc. Computer Science — MIT (2014) | GPA: 4.0/4.0"),
        hr(),
        h2("Publications"),
        li("'Cross-lingual Retrieval for Low-Resource Languages' — ACL 2022"),
        li("'Efficient Semantic Search with Compressed Embeddings' — EMNLP 2020"),
    ])
    print("OK cv_sarah_chen.pdf")


# ─── CV 3: Ahmed Hassan ──────────────────────────────────────────────────────
def cv_ahmed_hassan():
    doc = _doc('cv_ahmed_hassan.pdf')
    doc.build([
        h1("Ahmed Hassan"),
        p("ahmed.hassan@email.com | +49-151-555-0303 | Berlin, Germany"),
        hr(),
        h2("Profile"),
        p("DevOps and Platform Engineer with 6 years of experience designing CI/CD pipelines, "
          "Kubernetes clusters, and cloud infrastructure serving 100+ microservices. Expert in Docker, "
          "Kubernetes, Terraform, and AWS. Bilingual Arabic–English speaker."),
        hr(),
        h2("Technical Skills"),
        li("Containers / Orchestration: Docker, Kubernetes (K8s), Helm, Kustomize, K3s"),
        li("CI/CD: GitHub Actions, GitLab CI, Jenkins, ArgoCD, Tekton"),
        li("Infrastructure as Code: Terraform, Ansible, Pulumi, AWS CloudFormation"),
        li("Cloud: AWS (EKS, ECS, EC2, RDS, VPC, IAM, S3, CloudWatch), GCP (GKE)"),
        li("Observability: Prometheus, Grafana, Loki, Datadog, PagerDuty, Jaeger (distributed tracing)"),
        li("Languages / Scripting: Bash, Python, Go (basic), YAML"),
        li("Security: Vault (HashiCorp), RBAC, network policies, SAST/DAST integration"),
        hr(),
        h2("Work Experience"),
        h2("Senior Platform Engineer — CloudNative GmbH, Berlin (Sep 2021 – Present)"),
        li("Architected and managed AWS EKS cluster of 120+ nodes running 300+ microservices for a FinTech platform."),
        li("Reduced deployment time from 45 minutes to 6 minutes with GitHub Actions + ArgoCD (GitOps)."),
        li("Implemented Terraform modules managing 4 AWS accounts; eliminated manual provisioning drift."),
        li("Built Prometheus + Grafana observability stack with 200+ dashboards; reduced MTTR by 40%."),
        li("Led zero-downtime Kubernetes 1.25 → 1.28 cluster upgrade affecting 80+ production workloads."),
        li("Introduced Vault for secrets management, replacing hard-coded credentials across 60 services."),
        h2("DevOps Engineer — MENA Tech Solutions, Cairo (Jan 2018 – Aug 2021)"),
        li("Containerised 25 legacy applications with Docker; cut server costs by 30%."),
        li("Set up Jenkins CI/CD pipelines for 40+ repositories with SAST scanning (SonarQube)."),
        li("Managed on-premises VMware and led gradual migration to AWS EC2/RDS over 18 months."),
        li("Wrote Python automation scripts for backup, log rotation, and server health monitoring."),
        hr(),
        h2("Education"),
        p("B.Sc. Computer Engineering — Cairo University (2017) | GPA: 3.7/4.0"),
        hr(),
        h2("Certifications"),
        li("Certified Kubernetes Administrator (CKA) — CNCF, 2022"),
        li("AWS DevOps Engineer Professional — Amazon, 2021"),
        li("HashiCorp Certified: Terraform Associate — 2020"),
        hr(),
        h2("Languages"),
        p("Arabic (native) | English (fluent) | German (B1)"),
    ])
    print("OK cv_ahmed_hassan.pdf")


# ─── CV 4: Maria Garcia ───────────────────────────────────────────────────────
def cv_maria_garcia():
    doc = _doc('cv_maria_garcia.pdf')
    doc.build([
        h1("Maria Garcia"),
        p("maria.garcia@email.com | +34-612-555-0404 | Madrid, Spain | github.com/mgarcia-dev"),
        hr(),
        h2("About Me"),
        p("Frontend-focused Full-Stack Developer with 5 years of experience building responsive, "
          "accessible, and performant web applications with React and TypeScript. Comfortable across "
          "the full stack — from REST API design (Node.js / FastAPI) to UI component systems and CI/CD pipelines."),
        hr(),
        h2("Skills"),
        li("Frontend: React.js, TypeScript, Next.js, Vue.js 3, HTML5, CSS3, Tailwind CSS, Storybook"),
        li("State Management: Redux Toolkit, Zustand, React Query (TanStack Query), Context API"),
        li("Testing: Jest, React Testing Library, Cypress (E2E), Playwright"),
        li("Backend: Node.js, Express.js, FastAPI (Python), REST, GraphQL"),
        li("Databases: PostgreSQL, MongoDB, Firebase Realtime DB"),
        li("DevOps / Tools: Docker, GitHub Actions, Vercel, Netlify, Webpack, Vite"),
        li("Design: Figma, Adobe XD — comfortable working directly from design files"),
        hr(),
        h2("Work Experience"),
        h2("Senior Frontend Developer — DigitalProduct Studio, Madrid (Apr 2022 – Present)"),
        li("Led frontend architecture for SaaS project management platform (50k+ users) with React 18 + TypeScript + Next.js."),
        li("Built internal component library (60+ components) with Storybook, adopted across 5 product squads."),
        li("Improved Lighthouse performance score from 61 to 94 via code splitting, lazy loading, WebP/AVIF."),
        li("Introduced Cypress E2E suite covering 80% of critical flows; reduced regression reports by 45%."),
        li("Mentored 2 junior developers through pair programming and weekly 1:1 sessions."),
        h2("Full-Stack Developer — EdTech Startup, Remote (Jan 2019 – Mar 2022)"),
        li("Built interactive online learning platform (React + FastAPI) for 10,000 students across 3 countries."),
        li("Developed real-time collaborative whiteboard using WebSockets (Socket.io) and Canvas API."),
        li("Designed GraphQL API replacing legacy REST; reduced client-side data over-fetching by 55%."),
        li("Integrated Stripe subscriptions and PayPal with SCA compliance for EU users."),
        hr(),
        h2("Education"),
        p("B.Sc. Computer Science — Universidad Politécnica de Madrid (2018) | GPA: 9.1/10"),
        hr(),
        h2("Certifications"),
        li("Meta Frontend Developer Professional Certificate — Coursera, 2022"),
        li("Google UX Design Certificate — Coursera, 2021"),
        hr(),
        h2("Languages"),
        p("Spanish (native) | English (C1 – IELTS 7.5) | Portuguese (conversational)"),
    ])
    print("OK cv_maria_garcia.pdf")


# ─── CV 5: Layla Mostafa ──────────────────────────────────────────────────────
def cv_layla_mostafa():
    doc = _doc('cv_layla_mostafa.pdf')
    doc.build([
        h1("Layla Mostafa"),
        p("layla.mostafa@email.com | +971-50-555-0505 | Dubai, UAE | linkedin.com/in/layla-mostafa"),
        hr(),
        h2("Summary"),
        p("Full Stack Developer with 4 years of experience building scalable web applications for "
          "e-commerce, fintech, and media sectors across the MENA region. Fluent in English and Arabic; "
          "experienced delivering bilingual (Arabic/English) products with RTL layout support. "
          "Proficient in Python, JavaScript, React, and Django."),
        hr(),
        h2("Technical Skills"),
        li("Frontend: React.js, Next.js, TypeScript, HTML5, CSS3, Tailwind CSS, RTL layout (Arabic support)"),
        li("Backend: Python, Django, Django REST Framework, FastAPI, Node.js, Express"),
        li("Databases: PostgreSQL, MySQL, Redis, MongoDB"),
        li("DevOps: Docker, Docker Compose, GitHub Actions, Nginx, Linux server administration"),
        li("Cloud: AWS (EC2, S3, RDS), Firebase, Heroku"),
        li("Arabic NLP: arabic-reshaper, python-bidi, RTL text rendering, Arabic form validation"),
        li("Languages: Python (expert), JavaScript/TypeScript (expert), SQL"),
        hr(),
        h2("Experience"),
        h2("Full Stack Developer — Gulf Digital Solutions, Dubai (Feb 2022 – Present)"),
        li("Built bilingual (Arabic/English) e-commerce platform (Django + React) serving 120,000 MAU across 6 Gulf countries."),
        li("Implemented RTL-first CSS architecture and Arabic text normalisation; achieved WCAG 2.1 AA accessibility compliance."),
        li("Designed multi-currency payment integration (PayTabs, Telr) for AED, SAR, KWD, and EGP."),
        li("Reduced page load time 35% via Django caching (Redis), query optimisation, and CDN (CloudFront)."),
        li("Led API versioning and OpenAPI/Swagger documentation; onboarded 3 third-party partners."),
        h2("Junior Developer — MediaTech Beirut (Jul 2020 – Jan 2022)"),
        li("Developed CMS features for Arabic-language news portal (Django) with 500k daily readers."),
        li("Built React-based interactive data visualisation for election result dashboards."),
        li("Wrote Python scraping scripts (BeautifulSoup, Scrapy) from 40+ Arabic news sources."),
        hr(),
        h2("Education"),
        p("B.Sc. Computer Science — American University of Beirut (2020) | GPA: 3.75/4.0"),
        hr(),
        h2("Languages"),
        p("Arabic (native) | English (fluent) | French (intermediate)"),
    ])
    print("OK cv_layla_mostafa.pdf")


# ─── CV 6: Omar Abdullah (Arabic) ────────────────────────────────────────────
def cv_omar_arabic():
    """Arabic CV using arabic_reshaper + bidi for RTL rendering."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        print("WARN: arabic-reshaper / python-bidi not installed — run:  pip install arabic-reshaper python-bidi")
        return

    # Find an Arabic-capable TrueType font on the host machine
    font_candidates = [
        r"C:\Windows\Fonts\arial.ttf",                              # Windows
        r"C:\Windows\Fonts\ARIAL.TTF",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # Ubuntu/Debian
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",                   # Fedora/CentOS
        "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",  # macOS
        "/Library/Fonts/Arial Unicode MS.ttf",
    ]
    arabic_font_path = next((f for f in font_candidates if os.path.exists(f)), None)

    if not arabic_font_path:
        print("WARN: No Arabic-capable font found on this system — skipping cv_omar_arabic.pdf")
        print("   The existing cv_omar_arabic.html in data/raw/ will be used instead.")
        return

    pdfmetrics.registerFont(TTFont('ArabicFont', arabic_font_path))

    AR_H1   = ParagraphStyle('ArH1',   fontName='ArabicFont', fontSize=18, spaceAfter=6,  alignment=TA_RIGHT)
    AR_H2   = ParagraphStyle('ArH2',   fontName='ArabicFont', fontSize=12, spaceAfter=4,  spaceBefore=10, alignment=TA_RIGHT)
    AR_BODY = ParagraphStyle('ArBody', fontName='ArabicFont', fontSize=10, spaceAfter=4,  leading=14, alignment=TA_RIGHT)

    def ar(text, style=None):
        reshaped = arabic_reshaper.reshape(text)
        visual   = get_display(reshaped)
        return Paragraph(visual, style or AR_BODY)

    doc = _doc('cv_omar_arabic.pdf')
    doc.build([
        ar("عمر عبدالله", AR_H1),
        ar("omar.abdullah@email.com | القاهرة، مصر", AR_BODY),
        hr(),
        ar("الملخص المهني", AR_H2),
        ar("مطور ويب متكامل بخبرة 5 سنوات في تطوير تطبيقات الويب وقواعد البيانات. "
           "متخصص في React.js وNode.js وPython وقواعد بيانات PostgreSQL وMongoDB. "
           "أجيد اللغتين العربية والإنجليزية، وأمتلك خبرة في بناء منتجات رقمية للسوق العربي."),
        hr(),
        ar("المهارات التقنية", AR_H2),
        ar("• لغات البرمجة: Python، JavaScript، TypeScript، SQL"),
        ar("• الواجهة الأمامية: React.js، Next.js، HTML5، CSS3، Tailwind CSS"),
        ar("• الواجهة الخلفية: Node.js، Express.js، FastAPI، Django REST Framework"),
        ar("• قواعد البيانات: PostgreSQL، MongoDB، Redis، MySQL"),
        ar("• DevOps: Docker، Docker Compose، Git، GitHub Actions، Linux"),
        ar("• النصوص العربية: arabic-reshaper، python-bidi، تطبيع النصوص، استخراج النصوص من PDF"),
        hr(),
        ar("الخبرة العملية", AR_H2),
        ar("مطور ويب أول — شركة تقنيات رقمية للحلول، القاهرة (2021 – الحاضر)", AR_H2),
        ar("• قيادة فريق من 3 مطورين لبناء منصة تجارة إلكترونية تخدم 50,000 مستخدم يومياً."),
        ar("• تصميم وتطوير واجهات RESTful باستخدام FastAPI مع توثيق OpenAPI/Swagger."),
        ar("• تحسين أداء قاعدة بيانات PostgreSQL وتقليل وقت الاستجابة بنسبة 35%."),
        ar("• تطبيق نظام مصادقة JWT وOAuth2 مع تشفير البيانات الحساسة."),
        ar("مطور واجهة أمامية — مؤسسة الإبداع الرقمي، القاهرة (2019 – 2020)", AR_H2),
        ar("• بناء تطبيقات React.js للتعليم الإلكتروني تدعم العربية والإنجليزية مع تخطيط RTL كامل."),
        ar("• تطوير مكتبة مكونات UI مشتركة بين 5 مشاريع، مما قلص وقت التطوير بنسبة 40%."),
        hr(),
        ar("التعليم", AR_H2),
        ar("بكالوريوس هندسة الحاسبات والمعلومات — جامعة القاهرة (2019) | المعدل: 3.8/4.0"),
        hr(),
        ar("الشهادات المهنية", AR_H2),
        ar("• AWS Certified Developer Associate — 2023"),
        ar("• Meta Front-End Developer Professional Certificate — 2022"),
        ar("• Python for Data Science — IBM، 2021"),
        hr(),
        ar("اللغات", AR_H2),
        ar("العربية (اللغة الأم) | الإنجليزية (متقدم - IELTS 7.0) | الفرنسية (مبتدئ)"),
    ])
    print("OK cv_omar_arabic.pdf")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating PDFs -> {os.path.abspath(OUTPUT_DIR)}\n")
    cv_john_smith()
    cv_sarah_chen()
    cv_ahmed_hassan()
    cv_maria_garcia()
    cv_layla_mostafa()
    cv_omar_arabic()
    print("\nDone. Delete the .html files if you want PDF-only ingestion:")
    print(f"  del {os.path.abspath(OUTPUT_DIR)}\\*.html")
