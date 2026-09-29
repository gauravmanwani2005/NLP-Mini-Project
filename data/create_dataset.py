"""
Dataset generation and splitting script for EduMind.
Generates domain-specific academic queries across 12 intent categories,
and creates realistic academic policy documents for RAG.
"""

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

INTENTS = [
    "exam_schedule",
    "exam_registration",
    "attendance",
    "fees",
    "assignments",
    "results",
    "library",
    "placements",
    "courses",
    "timetable",
    "admissions",
    "general_query"
]

QUERY_TEMPLATES = {
    "exam_schedule": [
        "When are the semester {sem} examinations scheduled?",
        "What is the date sheet for {dept} {sem} semester finals?",
        "When will the mid-term exams begin this semester?",
        "Where can I find the final exam timetable?",
        "Is there any update on the upcoming university exams schedule?",
        "What dates are the practical exams being held?",
        "When do end-semester written exams start?",
        "Has the examination branch released the summer term date sheet?",
        "When is the supplementary examination scheduled?",
        "Could you tell me the timetable for {dept} theory exams?",
        "What time do morning session semester examinations begin?",
        "When will the viva voce schedule be announced for final year?",
        "Are exam dates postponed due to the national holiday?",
        "Where is the exam routine published on the college portal?",
        "What are the reporting timings for the final examinations?"
    ],
    "exam_registration": [
        "How do I submit the examination registration form?",
        "What is the deadline for filling the end-semester exam form?",
        "Can I register for back paper exams online?",
        "Where can I download my exam hall ticket or admit card?",
        "What is the late fee penalty for submitting the exam form late?",
        "Is exam form registration mandatory for regular students?",
        "How do I select backlog subjects during exam registration?",
        "My hall ticket shows an error in subject codes, whom to contact?",
        "What documents are needed to verify the exam application?",
        "When does the portal open for semester exam form submission?",
        "How to generate duplicate hall ticket if lost?",
        "Can I apply for improvement exams this semester?",
        "What is the exam application fee per subject?",
        "Where do I get the hall ticket signed by the HOD?",
        "How to verify if my exam registration status is confirmed?"
    ],
    "attendance": [
        "What is the minimum attendance percentage required for exams?",
        "Will I be detained if my attendance falls below 75 percent?",
        "How can I check my subject-wise attendance online?",
        "What is the medical leave attendance concession policy?",
        "Can duty leave for sports or cultural events compensate attendance?",
        "Who can condone attendance shortage between 65% and 75%?",
        "What happens if attendance is lower than the mandatory threshold?",
        "How often is the attendance record updated on the student ERP?",
        "Is 80% attendance compulsory for laboratory practical classes?",
        "How to apply for medical certificate submission for low attendance?",
        "Are there attendance penalties or debarment lists published?",
        "Does attendance carry internal assessment marks?",
        "Can HOD approve attendance relaxation for internship?",
        "What is the procedure for attendance grievance redressal?",
        "Where do I see my monthly attendance report?"
    ],
    "fees": [
        "What is the last date to pay the semester tuition fees?",
        "How do I pay academic fees through the online portal?",
        "What are the payment options for tuition and development fees?",
        "Is there a penalty or fine for delayed fee payment?",
        "Can semester fees be paid in multiple installments?",
        "How do I get an official fee receipt and payment acknowledgement?",
        "What are the hostel and mess fee charges for this year?",
        "Are there any scholarship schemes for tuition fee waivers?",
        "Whom should I contact in the accounts department for fee issues?",
        "What is the fee refund policy in case of course withdrawal?",
        "How to apply for post-matric government scholarship fee reimbursement?",
        "Where can I find the detailed fee breakdown for {dept}?",
        "Is there an educational loan assistance desk at the college?",
        "What is the laboratory caution deposit refund procedure?",
        "Are there concession benefits for merit students in fees?"
    ],
    "assignments": [
        "What is the submission deadline for semester assignments?",
        "How much weightage do internal assignments carry in total marks?",
        "Where do we need to upload project reports and assignments?",
        "What is the college policy on plagiarism in assignment submissions?",
        "Can I request an extension for mini-project report submission?",
        "How are continuous assessment tests and assignments evaluated?",
        "What format is required for submitting lab assignment records?",
        "Will late assignment submissions be penalized or rejected?",
        "Where can I check marks awarded for submitted assignments?",
        "Are handwritten or typed assignment reports accepted by faculty?",
        "How many internal assignments are compulsory per subject?",
        "What happens if I fail to submit the course assignment on time?",
        "Whom do I approach for clarification on assignment problem statements?",
        "Is peer review required for group assignment evaluations?",
        "Can re-submission be allowed if assignment marks are low?"
    ],
    "results": [
        "When will the odd semester results be declared?",
        "How do I check my semester grade card and SGPA/CGPA online?",
        "What is the procedure to apply for revaluation of exam papers?",
        "How to apply for photocopy or verification of answer scripts?",
        "What is the passing criteria and minimum grade required?",
        "When are the supplementary / backlog exam results announced?",
        "How can I apply for an official transcript for higher studies?",
        "What should I do if my result shows absent despite writing the exam?",
        "When will original grade sheets be distributed from the exam cell?",
        "How is CGPA calculated from individual semester SGPA scores?",
        "What are the grading brackets for letter grades O, A+, A, B?",
        "Is there a fee for requesting re-checking or re-totalling of marks?",
        "How long does the revaluation result take to be declared?",
        "What is the distinction criteria for honors degree completion?",
        "Can I get a provisional degree certificate after final results?"
    ],
    "library": [
        "What are the library opening and closing working hours?",
        "How many books can an undergraduate student borrow at a time?",
        "What is the loan duration and renewal period for library books?",
        "What is the late return fine charged per day per book?",
        "How do I access IEEE, Springer, and ScienceDirect digital journals?",
        "Is the digital library open on weekends and exam periods?",
        "How do I apply for a replacement if a borrowed book is lost?",
        "Can alumni and external researchers access the college library?",
        "Where can I find previous years university question papers in library?",
        "How to activate RemoteXs remote access for electronic research papers?",
        "Is there a dedicated quiet study or discussion room in the library?",
        "What are the photocopying and printing facilities inside library?",
        "How do I get a library clearance No-Dues certificate for graduation?",
        "Does the library provide plagiarism check via Turnitin for thesis?",
        "Can students recommend new reference textbooks for procurement?"
    ],
    "placements": [
        "What is the eligibility criteria for campus placement drives?",
        "Which top companies visit the college for campus recruitment?",
        "What is the minimum CGPA required to sit for tier-1 IT companies?",
        "Can students with active backlogs attend campus interviews?",
        "How do I register with the Training and Placement Cell (T&P)?",
        "Does the college provide mock interviews and aptitude training?",
        "What was the highest and average salary package offered last year?",
        "Is there a one-student-one-offer policy during placement season?",
        "What documents are needed during on-campus recruitment drives?",
        "How to apply for summer internships through the placement cell?",
        "Are core engineering companies recruiting from our department?",
        "What is the placement schedule for final year {dept} students?",
        "How to get NOC from placement cell for off-campus internships?",
        "What soft skill workshops are organized prior to placements?",
        "Where are placement drive notifications and shortlists published?"
    ],
    "courses": [
        "What are the elective courses offered in semester {sem} for {dept}?",
        "How do I register for open electives and multidisciplinary subjects?",
        "What are the prerequisites for enrolling in Advanced Machine Learning?",
        "Where can I view the complete syllabus and credit scheme for my degree?",
        "Can I change my elective subject after the semester starts?",
        "How many total credits are required to complete the B.Tech degree?",
        "Are NPTEL or Coursera online MOOC certifications eligible for credit transfer?",
        "What is the curriculum structure for honors and minor specializations?",
        "Where can I find course outcomes and learning objectives for subjects?",
        "Who is the course coordinator for Cloud Computing and DevOps?",
        "Can I audit a course from another engineering department?",
        "What is the laboratory credit weightage in the revised syllabus?",
        "How does the choice-based credit system (CBCS) work here?",
        "What happens if I miss the course registration deadline?",
        "Are interdisciplinary capstone projects allowed in final year?"
    ],
    "timetable": [
        "Where can I download the latest class timetable for {dept} {sem}?",
        "What are the lecture hours and lab session slots for Section A?",
        "Has the timetable changed for the upcoming working week?",
        "What is the daily schedule for theory classes and lunch break?",
        "When are the tutorial batches held for Mathematics and Physics?",
        "Which room or lecture hall is assigned for {dept} third year?",
        "Are Saturday classes scheduled on the academic timetable?",
        "Where is the faculty cabin and office hours consultation timetable?",
        "How can I see my laboratory batch allotment and timings?",
        "Is there a revised timetable during internal assessment week?",
        "What time does the first morning lecture commence?",
        "Who prepares and updates the department master timetable?",
        "Where is the link to subscribe to the Google Calendar timetable?",
        "What is the schedule for seminar and mini-project review sessions?",
        "Are compensatory lectures scheduled for public holidays?"
    ],
    "admissions": [
        "What is the admission procedure for B.Tech first year programs?",
        "What entrance exams (MHT-CET, JEE Main) are accepted for admission?",
        "What is the cutoff rank and percentile for {dept} engineering?",
        "How do I apply for direct second year (DSE) diploma lateral entry?",
        "What original documents are required during admission document verification?",
        "Is there a management quota or institutional level round for admission?",
        "What is the cancellation procedure and fee refund timeline for admission?",
        "Are scholarship benefits provided for economically backward students during admission?",
        "What is the hostel accommodation allotment process for first year freshers?",
        "When will the CAP round central admission merit lists be released?",
        "How to verify eligibility for OBC, SC, ST category reservations?",
        "Is an anti-ragging affidavit mandatory during admission reporting?",
        "What is the total intake capacity for {dept} undergraduate program?",
        "Can NRI / PIO / Kashmiri migrant candidates apply for admissions?",
        "Where can I download the college admission prospectus and brochure?"
    ],
    "general_query": [
        "What are the contact details and email address of the college registrar?",
        "Where is the administrative office located on campus?",
        "What are the college working hours and inquiry counter timings?",
        "Is college bus and transport facility available across the city?",
        "How can I apply for a bonafide student certificate or college ID card?",
        "Who is the designated college anti-ragging officer and committee head?",
        "What medical and emergency healthcare facilities exist on campus?",
        "How can students book the college auditorium or seminar hall for events?",
        "Where can I get a railway student concession pass signed?",
        "What is the procedure to issue a character or transfer certificate?",
        "Is Wi-Fi internet access provided to students in the college campus?",
        "What cafeteria and food court options are available in college?",
        "Whom do I report lost property or identity cards to on campus?",
        "When is the annual college cultural and technical fest organized?",
        "Are there student gym and sports complex facilities available?"
    ]
}

DEPARTMENTS = ["Computer Science", "Information Technology", "AI & Data Science", "Electronics", "Mechanical"]
SEMESTERS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]

VARIATIONS = [
    lambda text: text,
    lambda text: text.lower(),
    lambda text: "Please tell me: " + text.lower(),
    lambda text: "Can you help me know " + text[0].lower() + text[1:],
    lambda text: "Kindly inform " + text[0].lower() + text[1:],
    lambda text: "Urgent query regarding this: " + text,
    lambda text: "Hi, " + text[0].lower() + text[1:] + " Thanks!",
    lambda text: text.replace("?", " please?"),
    lambda text: text.replace("exam", "examination"),
    lambda text: text.replace("timetable", "schedule")
]

def generate_queries():
    dataset_records = []
    
    for intent, templates in QUERY_TEMPLATES.items():
        intent_queries = set()
        for template in templates:
            for dept in DEPARTMENTS:
                for sem in SEMESTERS:
                    q = template.format(dept=dept, sem=sem)
                    intent_queries.add(q)
        
        query_list = list(intent_queries)
        random.shuffle(query_list)
        # Select base queries
        selected_bases = query_list[:40]
        
        # Expand using variation transformations
        for base in selected_bases:
            for transform in VARIATIONS:
                variant = transform(base)
                intent_queries.add(variant)
                
        # Sample ~140-150 samples per intent
        sampled = list(intent_queries)
        random.shuffle(sampled)
        final_for_intent = sampled[:145]
        
        for q in final_for_intent:
            dataset_records.append({"text": q.strip(), "intent": intent})
            
    df = pd.DataFrame(dataset_records)
    df = df.drop_duplicates(subset=["text"])
    return df

def generate_documents():
    docs = {
        "attendance_policy.txt": """
EDUMIND COLLEGE OF ENGINEERING & TECHNOLOGY
OFFICIAL REGULATION: ATTENDANCE POLICY AND DETENTION RULES
Document Code: ECET-REG-ATT-2026-v2

1. MINIMUM ATTENDANCE REQUIREMENT
1.1 Every student is required to maintain a minimum attendance of 75% in aggregate across all registered courses (theory, laboratory, tutorials, and project work) in each semester to be eligible to appear for the end-semester examinations.
1.2 For laboratory courses and practical workshops, a strict minimum of 80% physical attendance is mandatory due to hands-on equipment and safety compliance.
1.3 Daily attendance is recorded electronically on the EduMind Student ERP portal at the conclusion of every lecture period and synced daily at 6:00 PM.

2. CONDONATION OF SHORTAGE OF ATTENDANCE
2.1 Condonation of attendance shortage between 65% and 74% may be granted by the College Academic Committee and the Principal only on genuine medical grounds or approved representation in university-level sports, NCC, NSS, or collegiate technical events.
2.2 For medical condonation, students must submit a formal medical leave application along with a registered medical practitioner's certificate and hospital discharge summary to the Head of the Department (HOD) within three working days of resuming classes.
2.3 An administrative condonation processing fee of Rs. 500 per subject is applicable for approved condonations.
2.4 Attendance below 65% shall NOT be condoned under any circumstances, including severe medical illness.

3. DETENTION AND DEBARMENT PROCEDURE
3.1 Students whose overall attendance remains below 75% (or 65% with approved condonation) shall be officially classified as 'Detained' for that academic semester.
3.2 Detained students are debarred from appearing in the regular end-semester theory and practical examinations.
3.3 A detained student will receive an 'F' (Fail) grade and must re-register for the entire semester courses in the subsequent academic year by paying the requisite semester re-admission tuition fees.
3.4 The provisional detention warning list is published by each department on the 1st of every month, and the final detention list is displayed seven days prior to the commencement of end-semester examinations.

4. ATTENDANCE GRIEVANCE REDRESSAL
4.1 Any discrepancy in recorded attendance must be brought to the notice of the concerned course instructor and Academic Coordinator within 48 hours of posting on the ERP.
4.2 Unresolved disputes may be escalated in writing to the Student Attendance Grievance Redressal Committee headed by the Dean of Academic Affairs.
""",

        "examination_regulations.txt": """
EDUMIND COLLEGE OF ENGINEERING & TECHNOLOGY
OFFICIAL REGULATION: EXAMINATION RULES, EVALUATION & REVALUATION GUIDELINES
Document Code: ECET-EXAM-REG-2026

1. EXAMINATION STRUCTURE AND ELIGIBILITY
1.1 The college conducts two In-Semester Continuous Assessment Tests (CAT-1 and CAT-2) carrying 20% weightage each, and one Comprehensive End-Semester Examination (ESE) carrying 60% weightage.
1.2 To qualify for appearing in the End-Semester Examination, a candidate must have:
    a) Maintained minimum 75% attendance in theory and 80% in laboratories.
    b) Cleared all tuition, laboratory, and library dues.
    c) Submitted all term-work assignments, mini-projects, and laboratory records certified by the faculty.
1.3 The morning session examination operates from 9:30 AM to 12:30 PM, and the afternoon session operates from 2:00 PM to 5:00 PM. Candidates must report at least 30 minutes before exam commencement.

2. EXAMINATION REGISTRATION AND HALL TICKETS
2.1 All eligible students must register online via the Examination Portal (exams.edumind.ac.in) for regular and backlog subjects at least 21 days before the scheduled exam date.
2.2 The regular examination registration fee is Rs. 1,500 per semester. Late registration with penalty of Rs. 500 is permitted up to 7 days after the regular deadline.
2.3 Hall Tickets / Admit Cards can be downloaded 5 days prior to exams after HOD online endorsement.
2.4 Carrying the printed Hall Ticket and valid College Identity Card inside the examination hall is strictly compulsory. Electronic gadgets, smartwatches, and programmable calculators are strictly prohibited.

3. PASSING CRITERIA AND GRADING SYSTEM
3.1 A student must secure a minimum of 40% marks in the End-Semester Examination and an aggregate of 45% (CAT + ESE) to obtain a passing grade in each subject.
3.2 Absolute 10-point Letter Grading Scale:
    - O (Outstanding): 90% and above (Grade Point: 10)
    - A+ (Excellent): 80% to 89% (Grade Point: 9)
    - A (Very Good): 70% to 79% (Grade Point: 8)
    - B+ (Good): 60% to 69% (Grade Point: 7)
    - B (Above Average): 50% to 59% (Grade Point: 6)
    - C (Pass): 45% to 49% (Grade Point: 5)
    - F (Fail): Below 45% (Grade Point: 0)
3.3 Cumulative Grade Point Average (CGPA) is computed as the weighted credit average of all semesters completed.

4. REVALUATION, RE-TOTALING AND PHOTOCOPY OF SCRIPTS
4.1 Students dissatisfied with their results may apply for Photocopy of Answer Scripts within 7 days of result declaration by paying Rs. 300 per subject.
4.2 Revaluation and Re-totaling applications must be submitted within 10 days of receiving the photocopy, accompanied by a fee of Rs. 700 per subject.
4.3 If revaluation results in a mark increase greater than 10% of maximum marks, the revised score replaces the original score, and 50% of the revaluation fee is refunded to the student.
4.4 Revaluation results are published within 25 working days from the application deadline.
""",

        "fee_structure_and_scholarships.txt": """
EDUMIND COLLEGE OF ENGINEERING & TECHNOLOGY
OFFICIAL REGULATION: TUITION FEES, HOSTEL DUES & SCHOLARSHIP SCHEMES
Document Code: ECET-FIN-SCH-2026

1. ANNUAL TUITION AND ACADEMIC FEES
1.1 The annual tuition fee for B.Tech Undergraduate engineering programs is Rs. 1,45,000 per academic year, payable in two equal semester installments.
1.2 Breakdown of Semester Fees:
    - Tuition Fee: Rs. 60,000 per semester
    - Development and Infrastructure Fee: Rs. 10,000 per semester
    - Laboratory and Computer Network Fee: Rs. 5,000 per semester
    - Library & Digital Resources Subscription: Rs. 2,500 per semester
    - Student Gymkhana, Sports & Cultural Fee: Rs. 2,500 per semester
    - Refundable Caution Deposit (One-time at admission): Rs. 10,000
1.3 All fees must be paid online via net banking, UPI, or debit/credit card through the EduMind ERP Fee Gateway. Cash payments are not accepted.

2. FEE PAYMENT DEADLINES AND PENALTIES
2.1 Odd Semester tuition fee must be cleared on or before July 31st.
2.2 Even Semester tuition fee must be cleared on or before December 31st.
2.3 A late payment penalty of Rs. 100 per day is levied for payments received between 1 and 15 days past deadline.
2.4 Accounts showing dues past 30 days will experience temporary ERP account suspension and exam hall ticket withholding.
2.5 Fee payment in installments requires prior written approval from the Registrar and Finance Officer on financial hardship demonstration.

3. HOSTEL AND MESS FEES
3.1 Triple-occupancy room with attached washroom: Rs. 45,000 per semester.
3.2 Double-occupancy AC room: Rs. 65,000 per semester.
3.3 Annual refundable hostel caution deposit: Rs. 8,000.
3.4 Mandatory cooperative mess subscription: Rs. 24,000 per semester (4 meals daily, vegetarian and non-vegetarian options).

4. SCHOLARSHIP SCHEMES AND TUITION CONCESSIONS
4.1 EduMind Merit Scholarship: 50% tuition waiver awarded to students securing top 5 ranks in their department with CGPA >= 9.2.
4.2 Economically Weaker Section (EWS) Assistance: Up to 40% tuition concession for students with verified annual family income below Rs. 4,00,000.
4.3 Government Scholarships: The college nodal scholarship cell facilitates Maharashtra Post-Matric Scholarship (MahaDBT), Central Sector Scholarship, and AICTE Pragati Scholarship for girl students.
4.4 Sports Excellence Award: Full tuition fee waiver for students representing the state or country in recognized national/international sporting championships.
""",

        "placement_and_training_handbook.txt": """
EDUMIND COLLEGE OF ENGINEERING & TECHNOLOGY
TRAINING & PLACEMENT CELL (T&P) GUIDELINES & RECRUITMENT POLICY
Document Code: ECET-TNP-POL-2026

1. PLACEMENT ELIGIBILITY & REGISTRATION
1.1 All final-year students seeking campus recruitment assistance must register with the T&P Portal (placement.edumind.ac.in) by July 15th with verified academic credentials.
1.2 Minimum Academic Eligibility:
    - Tier-1 Product Companies (CTC > 12 LPA): Minimum CGPA 8.0 without any live or past standing backlogs.
    - Tier-2 Core and IT Service Companies (CTC 6 to 12 LPA): Minimum CGPA 7.0 with maximum 1 standing backlog.
    - Mass Recruiters / Consultancies (CTC 3.5 to 6 LPA): Minimum CGPA 6.0 with maximum 2 allowed backlogs.
1.3 Attendance in all pre-placement training modules (quantitative aptitude, logical reasoning, verbal communication, DSA coding bootcamps, and mock interviews) must be at least 85%.

2. CAMPUS RECRUITMENT POLICY (ONE STUDENT - ONE OFFER)
2.1 To ensure equitable opportunities, EduMind follows a modified 'One Student, One Job Offer' policy:
    - Once a student receives an offer in Tier-3 category (< 6 LPA), they are permitted to compete only for Tier-1 'Dream' companies offering 2x of their current offer.
    - An offer from a Tier-1 Dream company (> 12 LPA) is considered final, and the candidate is immediately de-registered from all further placement drives.
2.2 Accepting an on-campus placement offer requires signing the Letter of Intent (LOI) within 48 hours. Failing to honor an accepted offer leads to permanent debarment from campus recruitment services.

3. INTERNSHIP GUIDELINES & CREDITS
3.1 Eighth-semester students may undertake full-time 6-month industry internships with stipend >= Rs. 20,000 per month, subject to departmental approval and No Objection Certificate (NOC) from the T&P Cell.
3.2 A 6-month industry internship carries 12 academic credits in lieu of elective coursework and capstone project-2, evaluated via monthly faculty reviews and joint company mentor appraisal.

4. RECRUITMENT STATISTICS HIGHLIGHTS
4.1 Over 140 premier technology companies participate in campus placements annually, including Google, Microsoft, Amazon, TCS, Infosys, Cognizant, L&T, and Tata Motors.
4.2 Highest international package offered: Rs. 44.5 LPA. Highest domestic package: Rs. 32 LPA. Average engineering package across all branches: Rs. 7.8 LPA.
4.3 T&P Cell contact: placement@edumind.ac.in, Office Room #204, Administrative Block.
""",

        "library_and_digital_resources.txt": """
EDUMIND COLLEGE OF ENGINEERING & TECHNOLOGY
CENTRAL LIBRARY & INFORMATION RESOURCE CENTRE: USER MANUAL
Document Code: ECET-LIB-MAN-2026

1. GENERAL TIMINGS AND MEMBERSHIP
1.1 The Central Library operates from 8:00 AM to 9:00 PM on all working days (Monday through Saturday).
1.2 During mid-term and end-semester examination periods, the 24x7 Reading Hall remains open overnight for registered students.
1.3 All bona fide students, faculty, and research scholars are automatically enrolled as library members upon issuance of their Barcoded Smart Identity Card.

2. CIRCULATION AND BORROWING PRIVILEGES
2.1 Undergraduate (B.Tech) Students: May borrow up to 4 books simultaneously for a borrowing period of 14 calendar days.
2.2 Postgraduate (M.Tech) & Research Scholars: May borrow up to 6 books for a duration of 28 calendar days.
2.3 Book renewals can be executed twice online via the Web-OPAC system, provided no reservation hold exists from another user.
2.4 Overdue Fine Structure:
    - Days 1 to 7 overdue: Rs. 5 per book per day.
    - Beyond 7 days overdue: Rs. 15 per book per day.
2.5 In case of lost or irreparably damaged books, the borrower must replace the book with the latest edition or pay three times the current catalogue market price plus Rs. 100 processing surcharge.

3. DIGITAL LIBRARY & E-RESOURCES
3.1 The Digital Library section houses 60 high-performance computer terminals equipped with high-speed internet and research tools.
3.2 Subscribed Online Databases & Electronic Journals:
    - IEEE Xplore Digital Library (Full-text access to all conferences and journals)
    - ACM Digital Library & SpringerLink Nature Journals
    - ScienceDirect (Elsevier) & ASCE / ASME Digital Collections
    - NPTEL video lecture repositories and Swayam local cache servers
3.3 RemoteXs Off-Campus Remote Access: Students can access digital databases from outside the campus using institutional email SSO credentials at elibrary.edumind.ac.in.
3.4 Anti-Plagiarism Screening: The library provides Turnitin and Urkund originality similarity verification for student project reports, thesis dissertations, and technical papers. A similarity index below 15% is mandatory for thesis submission.

4. CODE OF CONDUCT
4.1 Silence must be strictly observed in reading halls. Mobile phone usage in voice mode is strictly banned inside library premises.
4.2 Bags, personal books, and umbrellas must be deposited at the property counter before entering.
4.3 Library 'No-Dues Clearance' certificate is mandatory for final semester graduation and issuance of degree transcripts.
""",

        "academic_calendar_and_courses.txt": """
EDUMIND COLLEGE OF ENGINEERING & TECHNOLOGY
ACADEMIC CALENDAR, CURRICULUM SCHEME & TIMETABLE REGULATIONS
Document Code: ECET-ACAD-CAL-2026

1. ACADEMIC CALENDAR OVERVIEW
1.1 The academic year comprises two semesters: Odd Semester (July to December) and Even Semester (January to June).
1.2 Each semester consists of a minimum of 90 instructional days (15 teaching weeks) excluding examination and preparation breaks.
1.3 Important Dates for the Current Academic Year:
    - Odd Semester Commencement: July 15th
    - Continuous Assessment Test-1 (CAT-1): September 10th to 16th
    - Mid-Semester Technical Fest & Cultural Days: October 3rd to 5th
    - Continuous Assessment Test-2 (CAT-2): November 4th to 10th
    - Practical & Oral Examination Sessions: November 20th to 28th
    - End-Semester Theory Examinations: December 2nd to December 22nd
    - Semester Vacation: December 23rd to January 9th
    - Even Semester Classes Begin: January 10th

2. CHOICE-BASED CREDIT SYSTEM (CBCS) & COURSE REGISTRATION
2.1 B.Tech degree completion requires a cumulative earning of 160 academic credits distributed across:
    - Basic Science Courses (BSC): 24 Credits
    - Engineering Science Courses (ESC): 22 Credits
    - Professional Core Courses (PCC): 64 Credits
    - Professional Elective Courses (PEC): 18 Credits
    - Open Elective Courses (OEC): 12 Credits
    - Humanities & Social Sciences (HSMC): 12 Credits
    - Capstone Project Work & Internship: 16 Credits
2.2 Mandatory Course Registration takes place online in the first week of each semester. A student may register for a maximum of 28 credits per semester.
2.3 Adding or dropping elective courses is allowed within 10 instructional days of semester start with Academic Mentor recommendation.
2.4 Online MOOC credits: Up to 8 credits can be earned via approved NPTEL / SWAYAM 12-week certification courses with prior Dean approval.

3. MASTER TIMETABLE & WORKING HOURS
3.1 Instructional hours operate from 9:00 AM to 4:45 PM, Monday through Friday, divided into 7 lecture periods of 50 minutes each and a 55-minute lunch break (12:40 PM - 1:35 PM).
3.2 Laboratory sessions are conducted in consecutive 2-hour practical slots for designated batches of 20 students.
3.3 Saturday schedules are designated for co-curricular clubs, remedial tutorial sessions, NSS/sports activities, and student mentor meetings.
3.4 Official notifications, class schedules, and room allocations are published on the department notice boards and EduMind Student Mobile App.
"""
    }
    
    os.makedirs("data/raw/documents", exist_ok=True)
    for filename, content in docs.items():
        filepath = os.path.join("data/raw/documents", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"Created academic document: {filepath}")

def main():
    print("Generating academic documents...")
    generate_documents()
    
    print("Generating domain-specific queries...")
    df = generate_queries()
    print(f"Total unique queries generated: {len(df)}")
    print(f"Class distribution:\n{df['intent'].value_counts()}")
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    raw_path = "data/raw/college_queries.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved raw dataset to {raw_path}")
    
    # Stratified split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_SEED, stratify=df["intent"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["intent"]
    )
    
    train_path = "data/processed/train.csv"
    val_path = "data/processed/validation.csv"
    test_path = "data/processed/test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train split saved to {train_path} ({len(train_df)} rows)")
    print(f"Validation split saved to {val_path} ({len(val_df)} rows)")
    print(f"Test split saved to {test_path} ({len(test_df)} rows)")

if __name__ == "__main__":
    main()
