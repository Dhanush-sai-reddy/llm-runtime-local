import sqlite3
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_db():
    connection=sqlite3.connect("company.db")
    cursor=connection.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        role TEXT,
        department TEXT,
        join_date TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS performance (
        emp_id INTEGER,
        rating INTEGER,
        last_review_date TEXT,
        FOREIGN KEY(emp_id) REFERENCES employees(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS leave_balances (
        emp_id INTEGER,
        sick_leave_days INTEGER,
        vacation_days INTEGER,
        FOREIGN KEY(emp_id) REFERENCES employees(id)
    )
    """)

    employees=[
        (101, "Alice Carter", "Senior Dev", "Engineering", "2020-01-15"),
        (102, "Bob Smith", "Junior Analyst", "Marketing", "2023-06-01"),
        (103, "Charlie Davis", "Sales Manager", "Sales", "2019-03-10")
    ]
    
    performance=[(101, 5, "2024-12-01"), (102, 3, "2024-11-15"), (103, 2, "2024-10-20")]
    leaves=[(101, 5, 25), (102, 2, 0), (103, 10, 15)]

    cursor.executemany("INSERT OR IGNORE INTO employees VALUES (?,?,?,?,?)", employees)
    cursor.executemany("INSERT OR IGNORE INTO performance VALUES (?,?,?)", performance)
    cursor.executemany("INSERT OR IGNORE INTO leave_balances VALUES (?,?,?)", leaves)

    connection.commit()
    connection.close()
    print("✅ Database created.")

def create_pdf():
    c=canvas.Canvas("employee_handbook.pdf", pagesize=letter)
    width,height=letter
    y=height-50

    def check_page_break():
        nonlocal y
        if y<50:
            c.showPage()
            y=height-50

    def write(text, bold=False):
        nonlocal y
        check_page_break()
        if bold:
            c.setFont("Helvetica-Bold", 12)
            y-=10
        else:
            c.setFont("Helvetica", 10)
        
        c.drawString(50, y, text)
        y-=15

    write("Globex Corp - Comprehensive Employee Handbook 2025", bold=True)
    write("Effective Date: January 1st, 2025")
    y-=10

    write("1. Leave & Time Off Policy", bold=True)
    write("- Standard Vacation: All full-time employees receive 20 days per year.")
    write("- Sick Leave: 10 days per year. Doctor's note required after 3 consecutive days.")
    write("- Bereavement: 3 days paid leave for immediate family members.")
    write("- Unpaid Leave: Requires explicit written approval from the Department Manager.")
    write("- Carry-Over: Max 5 vacation days can carry over to next year. Sick days do not carry over.")

    write("2. Performance & Compensation", bold=True)
    write("- Performance Cycle: Reviews are conducted annually in December.")
    write("- Rating 5 (Star Performer): Eligible for 15% salary bonus + Stock Options.")
    write("- Rating 4 (Exceeds Expectations): Eligible for 5% salary bonus.")
    write("- Rating 3 (Meets Expectations): No bonus, but eligible for cost-of-living adjustment.")
    write("- Rating 1-2 (Needs Improvement): Mandatory Performance Improvement Plan (PIP).")

    write("3. Hybrid & Remote Work", bold=True)
    write("- Engineering Team: Fully remote allowed. Must reside in the same time zone +/- 3 hours.")
    write("- Sales Team: Hybrid model. Must be in the office Mon, Wed, Fri.")
    write("- Core Hours: All remote employees must be available between 10:00 AM and 3:00 PM EST.")
    write("- Equipment: Company provides one laptop and one monitor. Must be returned upon exit.")

    write("4. Health & Wellness Benefits", bold=True)
    write("- Medical Insurance: Covered 80% by Globex, 20% employee contribution.")
    write("- Dental/Vision: Covered 100% for employee, 50% for dependents.")
    write("- Gym Stipend: $50/month reimbursement for gym memberships or fitness apps.")
    write("- Mental Health: 5 free therapy sessions per year via our partner provider.")

    write("5. Travel & Expenses", bold=True)
    write("- Air Travel: Economy class for domestic, Premium Economy for international flights > 8 hours.")
    write("- Meals: Per diem of $75/day while traveling for business.")
    write("- Mileage: Reimbursed at the standard IRS rate for personal car usage.")
    write("- Approval: Expenses over $500 require prior VP approval.")

    write("6. IT Security & Code of Conduct", bold=True)
    write("- Passwords: Must be changed every 90 days. 2FA is mandatory.")
    write("- Data Privacy: No company data allowed on personal USB drives.")
    write("- Dress Code: Casual for Engineering; Business Casual for Client-Facing roles.")
    write("- Anti-Harassment: Zero tolerance policy. Report incidents to HR immediately.")

    c.save()
    print("✅ Expanded PDF 'employee_handbook.pdf' created.")

if __name__=="__main__":
    create_db()
    create_pdf()