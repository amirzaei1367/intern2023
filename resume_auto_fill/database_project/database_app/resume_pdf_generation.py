from reportlab.lib.pagesizes import letter, inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Flowable
from reportlab.graphics.shapes import Line

# 1. Create a custom Flowable for the line.
class ColoredLine(Flowable):
    def __init__(self, width, color):
        Flowable.__init__(self)
        self.width = width
        self.color = color

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.line(0, 0, self.width, 0)

def create_resume_pdf(first_name, last_name, education, work_history, skills, certificates, security_clearance, file_path, logo_path):
    # Define custom styles
    styles = getSampleStyleSheet()

    header_style = styles['Heading1']
    header_style.fontName = 'Times-Roman' # <-- Updated here
    header_style.fontSize = 24
    header_style.leading = 28

    subheader_style = styles['Heading2']
    subheader_style.fontName = 'Times-Roman' # <-- Updated here
    subheader_style.fontSize = 16
    subheader_style.leading = 20
    subheader_style.textColor = colors.steelblue

    normal_style = styles['Normal']
    normal_style.fontName = 'Times-Roman' # <-- Updated here
    normal_style.fontSize = 12
    normal_style.leading = 16


    # Create a PDF document
    doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=72, leftMargin=72)
    content = []

    # Logo
    logo = Image(logo_path, width=1.5*inch, height=1.5*inch)

    # Name and initials
    header_text = f"{first_name} {last_name}"
    initials = first_name[0]+last_name[0]  # Use non-breaking space

    # Create initials in a blue square
    initials_style = ParagraphStyle('initials', fontName='Times-Bold', fontSize=32, leading=42, textColor=colors.whitesmoke, backColor=colors.steelblue, alignment=1, spaceAfter=6)
    initials_paragraph = Paragraph(initials, initials_style) # Ensuring font remains consistent and bold

    # Use a Table to align the initials, name, and logo
    header_table_data = [[initials_paragraph, Paragraph(header_text, header_style), logo]]
    header_table = Table(header_table_data, colWidths=[0.75*inch, 3*inch, 1.5*inch], rowHeights=[1*inch])
    header_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    content.append(header_table)
    content.append(Spacer(1, 24))

    available_width = letter[0] - 2*72  # subtracting margins
    separator_line = ColoredLine(available_width, colors.steelblue)

    # Helper function to add a section
    def add_section(title, text):
        content.append(Paragraph(title, subheader_style))
        content.append(Spacer(1, 5))
        content.append(Paragraph(text, normal_style))
        content.append(Spacer(1, 10))
        content.append(separator_line)
        content.append(Spacer(1, 16))

    # Sections
    add_section('Education', education)
    add_section('Work History', work_history)
    add_section('Skills', skills)
    add_section('Certificates', certificates)
    add_section('Security Clearance', security_clearance)

    # Build the PDF
    doc.build(content)

# Example usage
resume_path = "output.pdf"
first_name = "John"
last_name = "Doe"
education = "Bachelor of Science in Computer Science<br/>XYZ University<br/>Graduated: May 2022"
work_history = "Software Engineer<br/>XYZ Corp<br/>May 2022 - Present<br/>- Developed and maintained web applications<br/>- Collaborated with cross-functional teams"
skills = "Python, JavaScript, SQL, Data Analysis, Problem Solving"
certificates = "Certified Python Developer<br/>AWS Certified Developer"
security_clearance = "Top Secret"
logo_path = "3747.png"

create_resume_pdf(first_name, last_name, education, work_history, skills, certificates, security_clearance, resume_path, logo_path)
