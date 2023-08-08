import torch
import pandas as pd
import time
import os
import sys
import fire 
from utils import load_model, load_tokenizer
from dataclasses import dataclass
from utils.chat_utils import format_tokens 
from tqdm import tqdm 

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

schema_linking_prompt = '''Table advisor, columns = [*,s_ID,i_ID]
Table classroom, columns = [*,building,room_number,capacity]
Table course, columns = [*,course_id,title,dept_name,credits]
Table department, columns = [*,dept_name,building,budget]
Table instructor, columns = [*,ID,name,dept_name,salary]
Table prereq, columns = [*,course_id,prereq_id]
Table section, columns = [*,course_id,sec_id,semester,year,building,room_number,time_slot_id]
Table student, columns = [*,ID,name,dept_name,tot_cred]
Table takes, columns = [*,ID,course_id,sec_id,semester,year,grade]
Table teaches, columns = [*,ID,course_id,sec_id,semester,year]
Table time_slot, columns = [*,time_slot_id,day,start_hr,start_min,end_hr,end_min]
Foreign_keys = [course.dept_name = department.dept_name,instructor.dept_name = department.dept_name,section.building = classroom.building,section.room_number = classroom.room_number,section.course_id = course.course_id,teaches.ID = instructor.ID,teaches.course_id = section.course_id,teaches.sec_id = section.sec_id,teaches.semester = section.semester,teaches.year = section.year,student.dept_name = department.dept_name,takes.ID = student.ID,takes.course_id = section.course_id,takes.sec_id = section.sec_id,takes.semester = section.semester,takes.year = section.year,advisor.s_ID = student.ID,advisor.i_ID = instructor.ID,prereq.prereq_id = course.course_id,prereq.course_id = course.course_id]
#Q: "Find the buildings which have rooms with capacity more than 50." [/INST]
#A: Let’s think step by step. In the question "Find the buildings which have rooms with capacity more than 50.", we are asked:
"the buildings which have rooms" so we need column = [classroom.capacity]
"rooms with capacity" so we need column = [classroom.building]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [50]. So the Schema_links are:
Schema_links: [classroom.building,classroom.capacity,50]

[INST] Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
#Q: "How many heads of the departments are older than 56 ?"[/INST]
#A: Let’s think step by step. In the question "How many heads of the departments are older than 56 ?", we are asked:
"How many heads of the departments" so we need column = [head.*]
"older" so we need column = [head.age]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [56]. So the Schema_links are:
Schema_links: [head.*,head.age,56]

[INST] Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
#Q: "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?"[/INST]
#A: Let’s think step by step. In the question "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?", we are asked:
"distinct creation years of the departments" so we need column = [department.Creation]
"departments managed by" so we need column = [management.department_ID]
"born in" so we need column = [head.born_state]
Based on the columns and tables, we need these Foreign_keys = [department.Department_ID = management.department_ID,management.head_ID = head.head_ID].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = ['Alabama']. So the Schema_links are:
Schema_links: [department.Creation,department.Department_ID = management.department_ID,head.head_ID = management.head_ID,head.born_state,'Alabama']
'''

#[INST]Table Addresses, columns = [*,address_id,line_1,line_2,city,zip_postcode,state_province_county,country]
#Table Candidate_Assessments, columns = [*,candidate_id,qualification,assessment_date,asessment_outcome_code]
#Table Candidates, columns = [*,candidate_id,candidate_details]
#Table Courses, columns = [*,course_id,course_name,course_description,other_details]
#Table People, columns = [*,person_id,first_name,middle_name,last_name,cell_mobile_number,email_address,login_name,password]
#Table People_Addresses, columns = [*,person_address_id,person_id,address_id,date_from,date_to]
#Table Student_Course_Attendance, columns = [*,student_id,course_id,date_of_attendance]
#Table Student_Course_Registrations, columns = [*,student_id,course_id,registration_date]
#Table Students, columns = [*,student_id,student_details]
#Foreign_keys = [Students.student_id = People.person_id,People_Addresses.address_id = Addresses.address_id,People_Addresses.person_id = People.person_id,Student_Course_Registrations.course_id = Courses.course_id,Student_Course_Registrations.student_id = Students.student_id,Student_Course_Attendance.student_id = Student_Course_Registrations.student_id,Student_Course_Attendance.course_id = Student_Course_Registrations.course_id,Candidates.candidate_id = People.person_id,Candidate_Assessments.candidate_id = Candidates.candidate_id]
#Q: "List the id of students who never attends courses?"[/INST]
#A: Let’s think step by step. In the question "List the id of students who never attends courses?", we are asked:
#"id of students" so we need column = [Students.student_id]
#"never attends courses" so we need column = [Student_Course_Attendance.student_id]
#Based on the columns and tables, we need these Foreign_keys = [Students.student_id = Student_Course_Attendance.student_id].
#Based on the tables, columns, and Foreign_keys, The set of possible cell values are = []. So the Schema_links are:
#Schema_links: [Students.student_id = Student_Course_Attendance.student_id]
#'''

classification_prompt = '''Q: "Find the buildings which have rooms with capacity more than 50."
schema_links: [classroom.building,classroom.capacity,50] [/INST]
A: Let’s think step by step. The SQL query for the question "Find the buildings which have rooms with capacity more than 50." needs these tables = [classroom], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

[INST] Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings." 
schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100] [/INST]
A: Let’s think step by step. The SQL query for the question "Find the room number of the rooms which can sit 50 to 100 students and their buildings." needs these tables = [classroom], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

[INST] Q: "How many courses that do not have prerequisite?"
schema_links: [course.*,course.course_id = prereq.course_id] [/INST] 
A: Let’s think step by step. The SQL query for the question "How many courses that do not have prerequisite?" needs these tables = [course,prereq], so we need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Which courses have prerequisite?"].
So, we need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"


[INST] Q: "What is the name of the instructor who advises the student with the greatest number of total credits?"
schema_links: [advisor.i_id = instructor.id,advisor.s_id = student.id,instructor.name,student.tot_cred ] [/INST] 
A: Let’s think step by step. The SQL query for the question "What is the name of the instructor who advises the student with the greatest number of total credits?" needs these tables = [advisor,instructor,student], so we need JOIN.
Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"
'''

#[INST] Q: "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010."
#schema_links: [teaches.id,teaches.semester,teaches.year,Fall,2009,Spring,2010] [/INST] 
#A: Let’s think step by step. The SQL query for the question "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010." needs these tables = [teaches], so we don't need JOIN.
#Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Find the id of instructors who taught a class in Spring 2010"].
#So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
#Label: "NESTED"

#[INST] Q: "Find the name of the department that offers the highest total credits?"
#schema_links: [course.dept_name,course.credits] [/INST]
#A: Let’s think step by step. The SQL query for the question "Find the name of the department that offers the highest total credits?." needs these tables = [course], so we don't need JOIN.
#Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
#So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
#Label: "EASY"

#[INST] Q: "What is the name of the instructor who advises the student with the greatest number of total credits?"
#schema_links: [advisor.i_id = instructor.id,advisor.s_id = student.id,instructor.name,student.tot_cred ] [/INST] 
#A: Let’s think step by step. The SQL query for the question "What is the name of the instructor who advises the student with the greatest number of total credits?" needs these tables = [advisor,instructor,student], so we need JOIN.
#Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
#So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
#Label: "NON-NESTED"

#[INST] Q: "Find the total number of students and total number of instructors for each department."
#schema_links = [department.dept_name = instructor.dept_name,student.id,student.dept_name = department.dept_name,instructor.id] [/INST] 
#A: Let’s think step by step. The SQL query for the question "Find the total number of students and total number of instructors for each department." needs these tables = [department,instructor,student], so we need JOIN.
#Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
#So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
#Label: "NON-NESTED"
#'''

easy_prompt = '''Q: "Find the buildings which have rooms with capacity more than 50."
Schema_links: [classroom.building,classroom.capacity,50] [/INST]
SQL: SELECT DISTINCT building FROM classroom WHERE capacity  >  50 

[INST] Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings."
Schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100] [/INST] 
SQL: SELECT building ,  room_number FROM classroom WHERE capacity BETWEEN 50 AND 100

[INST] Q: "Give the name of the student in the History department with the most credits."
Schema_links: [student.name,student.dept_name,student.tot_cred,History] [/INST]
SQL: SELECT name FROM student WHERE dept_name  =  'History' ORDER BY tot_cred DESC LIMIT 1

[INST] Q: "Find the total budgets of the Marketing or Finance department."
Schema_links: [department.budget,department.dept_name,Marketing,Finance] [/INST]
SQL: SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance' 

[INST] Q: "Find the department name of the instructor whose name contains 'Soisalon'."
Schema_links: [instructor.dept_name,instructor.name,Soisalon] [/INST]
SQL: SELECT dept_name FROM instructor WHERE name LIKE '%Soisalon%'
'''

#[INST] Q: "What is the name of the department with the most credits?"
#Schema_links: [course.dept_name,course.credits] [/INST] 
#SQL: SELECT dept_name FROM course GROUP BY dept_name ORDER BY sum(credits) DESC LIMIT 1

#[INST] Q: "How many instructors teach a course in the Spring of 2010?"
#Schema_links: [teaches.ID,teaches.semester,teaches.YEAR,Spring,2010] [/INST]
#SQL: SELECT COUNT (DISTINCT ID) FROM teaches WHERE semester  =  'Spring' AND YEAR  =  2010
#'''

#[INST] Q: "Find the name of the students and their department names sorted by their total credits in ascending order."
#Schema_links: [student.name,student.dept_name,student.tot_cred] [/INST]
#SQL: SELECT name ,  dept_name FROM student ORDER BY tot_cred
#'''

#Q: "Find the year which offers the largest number of courses."
#Schema_links: [SECTION.YEAR,SECTION.*]
#SQL: SELECT YEAR FROM SECTION GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1

#Q: "What are the names and average salaries for departments with average salary higher than 42000?"
#Schema_links: [instructor.dept_name,instructor.salary,42000]
#SQL: SELECT dept_name ,  AVG (salary) FROM instructor GROUP BY dept_name HAVING AVG (salary)  >  42000
#'''

medium_prompt = '''Q: "Find the total budgets of the Marketing or Finance department."
Schema_links: [department.budget,department.dept_name,Marketing,Finance] [/INST]
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select sum(department.budget) from department  where  department.dept_name = \"Marketing\"  or  department.dept_name = \"Finance\"
SQL: SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance'

[INST] Q: "Find the name and building of the department with the highest budget."
Schema_links: [department.budget,department.dept_name,department.building] [/INST] 
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select department.dept_name , department.building from department  order by department.budget desc limit 1
SQL: SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1

[INST] Q: "What is the name and building of the departments whose budget is more than the average budget?"
Schema_links: [department.budget,department.dept_name,department.building] [/INST] 
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation:  select department.dept_name , department.building from department  where  @.@ > avg ( department.budget ) 
SQL: SELECT dept_name ,  building FROM department WHERE budget  >  (SELECT avg(budget) FROM department)

[INST] Q: "Find the total number of students and total number of instructors for each department."
Schema_links: [department.dept_name = student.dept_name,student.id,department.dept_name = instructor.dept_name,instructor.id] [/INST] 
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = [department,student,instructor]. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: "select count( distinct student.ID) , count( distinct instructor.ID) , department.dept_name from department  group by instructor.dept_name
SQL: SELECT count(DISTINCT T2.id) ,  count(DISTINCT T3.id) ,  T3.dept_name FROM department AS T1 JOIN student AS T2 ON T1.dept_name  =  T2.dept_name JOIN instructor AS T3 ON T1.dept_name  =  T3.dept_name GROUP BY T3.dept_name
'''

hard_prompt = '''Q: "Find the title of courses that have two prerequisites?"
Schema_links: [course.title,course.course_id = prereq.course_id] [/INST] 
A: Let's think step by step. "Find the title of courses that have two prerequisites?" can be solved by knowing the answer to the following sub-question "What are the titles for courses with two prerequisites?".
The SQL query for the sub-question "What are the titles for courses with two prerequisites?" is SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2
So, the answer to the question "Find the title of courses that have two prerequisites?" is =
Intermediate_representation: select course.title from course  where  count ( prereq.* )  = 2  group by prereq.course_id
SQL: SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2

[INST] Q: "Find the name and building of the department with the highest budget."
Schema_links: [department.dept_name,department.building,department.budget] [/INST] 
A: Let's think step by step. "Find the name and building of the department with the highest budget." can be solved by knowing the answer to the following sub-question "What is the department name and corresponding building for the department with the greatest budget?".
The SQL query for the sub-question "What is the department name and corresponding building for the department with the greatest budget?" is SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1
So, the answer to the question "Find the name and building of the department with the highest budget." is =
Intermediate_representation: select department.dept_name , department.building from department  order by department.budget desc limit 1
SQL: SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1

[INST] Q: "Find the title, credit, and department name of courses that have more than one prerequisites?"
Schema_links: [course.title,course.credits,course.dept_name,course.course_id = prereq.course_id] [/INST]
A: Let's think step by step. "Find the title, credit, and department name of courses that have more than one prerequisites?" can be solved by knowing the answer to the following sub-question "What is the title, credit value, and department name for courses with more than one prerequisite?".
The SQL query for the sub-question "What is the title, credit value, and department name for courses with more than one prerequisite?" is SELECT T1.title ,  T1.credits , T1.dept_name FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  >  1
So, the answer to the question "Find the name and building of the department with the highest budget." is =
Intermediate_representation: select course.title , course.credits , course.dept_name from course  where  count ( prereq.* )  > 1  group by prereq.course_id 
SQL: SELECT T1.title ,  T1.credits , T1.dept_name FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  >  1
'''

#[INST] Q: "Give the name and building of the departments with greater than average budget." 
#Schema_links: [department.dept_name,department.building,department.budget] [/INST]
#A: Let's think step by step. "Give the name and building of the departments with greater than average budget." can be solved by knowing the answer to the following sub-question "What is the average budget of departments?".
#The SQL query for the sub-question "What is the average budget of departments?" is SELECT avg(budget) FROM department
#So, the answer to the question "Give the name and building of the departments with greater than average budget." is =
#Intermediate_representation: select department.dept_name , department.building from department  where  @.@ > avg ( department.budget )
#SQL: SELECT dept_name ,  building FROM department WHERE budget  >  (SELECT avg(budget) FROM department)
#'''
def load_dataset(DATASET):
    return pd.read_json(DATASET)

def hard_prompt_maker(spider_schema, spider_foreign, test_sample_text, database, schema_links, sub_questions):
    instruction = "[INST]" + "<<SYS>>\n" + "#Use the intermediate representation and the schema links to generate the SQL queries for each of the questions"
    fields = find_fields_SQL_like(spider_schema, "college_2")
    fields += "Foreign_keys = " + find_foreign_keys_SQL_like(spider_foreign, "college_2") + '\n'
    fields += find_fields_SQL_like(spider_schema, database)
    fields += "Foreign_keys = " + find_foreign_keys_SQL_like(spider_foreign, database) + '\n'
    stepping = f'''\nA: Let's think step by step. "{test_sample_text}" can be solved by knowing the answer to the following sub questions "{sub_questions}".'''
    fields += "\n"
    prompt = instruction + fields + E_SYS  + hard_prompt + "[INST]" +  'Q: "' + test_sample_text + '"' + '\nschema_links: '+ schema_links + "[/INST]" +  stepping + '\nThe SQL query for the sub-question'
    return prompt

def medium_prompt_maker(spider_schema, spider_foreign, test_sample_text, database, schema_links):
    instruction = B_INST + B_SYS + "# Use the schema links and Intermediate_representation to generate the SQL queries for each of the questions"
    fields = find_fields_SQL_like(spider_schema, "college2")
    fields += "Foreign_keys = " + find_foreign_keys_SQL_like(spider_foreign, "college_2") + '\n'
    fields += find_fields_SQL_like(spider_schema, database)
    fields += "Foreign_keys = " + find_foreign_keys_SQL_like(spider_foreign, database) + '\n'
    fields += "\n"
    prompt = instruction + fields + E_SYS + medium_prompt + 'Q: "' + test_sample_text + '\nSchema_links: ' + schema_links + E_INST + "\nA: Let's think step by step."
    return prompt

def easy_prompt_maker(spider_schema, test_sample_text, database, schema_links):
    instruction = B_INST + B_SYS + "# Use the schema links to generate the SQL queries for each of the questions."
    fields = find_fields_SQL_like(spider_schema, "college_2")
    fields += find_fields_SQL_like(spider_schema, database)
    fields += "\n"
    prompt = instruction + fields + E_SYS  + easy_prompt  + B_INST +  ' Q: "' + test_sample_text + '\nSchema_links: ' +   schema_links + ' ' + E_INST +  '\nSQL:'
    return prompt

def classification_prompt_maker(spider_schema, spider_foreign, test_sample_text, database, schema_links):
    instruction = B_INST + B_SYS + '# For the given question, classify it as EASY, NON-NESTED, or NESTED based on nested queries and JOIN.\n'
    instruction += "\nif need nested queries: predict NESTED\n"
    instruction += "elif need JOIN and don't need nested queries: predict NON-NESTED\n"
    instruction += "elif don't need JOIN and don't need nested queries: predict EASY\n\n" + E_SYS
    fields = find_fields_SQL_like(spider_schema, 'college_2')
    fields += "Foreign_keys = " + find_foreign_keys_SQL_like(spider_foreign, "college_2") + '\n'
    fields += find_fields_SQL_like(spider_schema, database)
    fields += "Foreign_keys = " + find_foreign_keys_SQL_like(spider_foreign, database) + '\n'
    fields += "\n"
    prompt = instruction + fields + classification_prompt + B_INST + ' Q: "' + test_sample_text + '\nschema_links: ' + schema_links + E_INST +  "\nA: Let's think step by step"
    return prompt

def schema_linking_prompt_maker(spider_schema, spider_foreign, test_sample_text, database):
    instruction = "[INST]" + "<<SYS>>\n" + "# Find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.\n" + "n<</SYS>>\n\n"
    fields = find_fields_SQL_like(spider_schema, database)
    foreign_keys = "Foreign_keys = " + find_foreign_keys_SQL_like(spider_foreign, database) + '\n'
    prompt = instruction + schema_linking_prompt + '[INST]' + fields + foreign_keys + 'Q: "' + test_sample_text + "[/INST]" + """\nA: Let's Think step by step."""
    return prompt

def find_fields_SQL_like(spider_schema, db_name):
    df = spider_schema[spider_schema['Database name']==db_name]
    df = df.groupby('Table Name')
    output = ""
    for name, group in df:
        output += "Table " + name+  ', columnns = ['
        for index, row in group.iterrows():
            output += row["Field Name"] + ','
        output = output[:-1]
        output += "]\n"
    return output

def find_foreign_keys_SQL_like(spider_foreign, db_name):
    df = spider_foreign[spider_foreign['Database name'] == db_name]
    output = "["
    for index, row in df.iterrows():
        output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + "=" + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
    output = output[:-1] + "]"
    return output

def find_primary_keys_SQL_like(spider_primary, db_name):
    df = spider_primary[spider_primary['Database name'] == db_name]
    output = '['
    for index, row in df.iterrows():
        output += row['Table Name'] + '.' + row['Primary Key'] + ','
    output = output[:-1]
    output += "]\n"
    return output

def creating_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names', 'table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])

        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', 'Table Name', 'Field Name', 'Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key', 'Second Table Foreign Key']
    )
    return spider_schema, spider_primary, spider_foreign

def debugger(spider_schema, spider_foreign, spider_primary, test_sample_text, database, sql):
    instruction = B_INST + B_SYS + """### For the given question, use the provided tables, columns foreign keys, and primary keys to fix the given SQLite SQL QUERY  for any issues.
    If there are any problems, fix them. If there are no issues, return the SQL QUERY as is.
    ### Use the following instructions for fixing the SQL Query:
    1) Use the database values that are explicitly mentioned in the question.
    2) Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
    3) Use DESC and DISTINCT when needed.
    4) Pay attention to the columns that are used for the GROUP BY statement.
    5) Pay attention to the columns that are used for the SELECT statement.
    6) Only change the GROUP BY clause when necessary (avoid redundant columns in GROUP BY)
    7) Use GROUP BY on one column only
    """  
    fields = find_fields_SQL_like(spider_schema, database)
    fields += "Foreign_keys = " + find_foreign_keys_SQL_like(spider_foreign, database) + '\n'
    fields += "Primary_keys = " + find_primary_keys_SQL_like(spider_primary, database) 
    prompt = instruction + fields + E_SYS + '### Question: ' + test_sample_text +  '\n#### SQLite SQL QUERY\n' + sql + E_INST +  '\n#### SQLite FIXED SQL QUERY\nSELECT'
    return prompt

def debugger2(spider_schema, spider_foreign, spider_primary, test_sample_text, database, sql):
    instruction = B_INST + B_SYS + """###Fix the bugs in the below SQL for the give question.
    """  
    fields = find_fields_SQL_like(spider_schema, database)
    fields += "Foreign_keys = " + find_foreign_keys_SQL_like(spider_foreign, database) + '\n'
    fields += "Primary_keys = " + find_primary_keys_SQL_like(spider_primary, database) 
    prompt = instruction + fields + E_SYS + '### Question: ' + test_sample_text +  '\n### BUGGY SQL\n' + sql + E_INST +  '\n#### SQLite FIXED SQL' 
    return prompt


def generation(tokens,model, tokenizer, max_new_tokens, temperature, top_p):
    with torch.no_grad():
        tokens = {k:v.to("cuda") for k,v in model_inputs.items()}
        outputs = model.generate(
            **tokens,
            max_new_tokens=128,
            temperature=temperature,
            top_p=top_p,
        )
        output_text = tokenizer.decode(outputs[0], skip_special_token=True)
        output_text = classification.strip()
    return output_text

def main(
    model_id: str='models_hf/7B-chat/',
    quantization: bool=True,
    dataset_schema: str='dataset/spider/tables.json',
    dataset: str='dataset/spider/dev.json',
    output_file: str='dinsql.csv',
    max_new_tokens: int=256,
    temperature: float=1.0,
    top_p: float=0.9,
    do_sample: bool=False,
):
    model = load_model(model_id, quantization)
    tokenizer = load_tokenizer(model_id)

    CODEX = []

    spider_schema, spider_primary, spider_foreign = creating_schema(dataset_schema)
    val_df = load_dataset(dataset)
    print(f"Number of data samples {val_df.shape[0]}")

    for index, row in val_df.iterrows():
        print(f"index is {index}")
        print(row['query'])
        print(row['question'])

        schema_links = None

        while schema_links is None:
            prompt = schema_linking_prompt_maker(spider_schema, spider_foreign, row['question'], row['db_id'])
            model_inputs = tokenizer(prompt, return_tensors='pt')
            model.eval()
            try: 
                with torch.no_grad():
                    tokens = {k:v.to("cuda") for k,v in model_inputs.items()}
                    outputs = model.generate(
                        **tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    schema_links = tokenizer.decode(outputs[0], skip_special_token=True)
                    schema_links = schema_links.strip()

            except:
                time.sleep(3)
                pass 

        try:
            schema_links = schema_links.split("Schema_links: ")
            schema_links = schema_links[-1]
        except:
            print("Slicing error for the schema_linking module")
            schema_links = "[]"

        classification = None
        while classification is None:
            prompt = classification_prompt_maker(spider_schema, spider_foreign, row['question'], row['db_id'], schema_links)
            #print(prompt)
            model_inputs = tokenizer(prompt, return_tensors='pt')
            model.eval()
            try:
                with torch.no_grad():
                    tokens = {k:v.to("cuda") for k,v in model_inputs.items()}
                    outputs = model.generate(
                        **tokens,
                        max_new_tokens=128,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=False,
                    )
                    classification = tokenizer.decode(outputs[0], skip_special_token=True)
                    classification = classification.strip()
                    #print(classification.split("Label: "))

            except:
                time.sleep(3)
                pass 
        try:
            predicted_class = classification.split("Label: ")[-1]
        except:
            print("Sclicing error for the classfication module")
            predicted_class = '"NESTED"'

        if '"EASY"' in predicted_class:
            print("EASY")
            SQL = None
            prompt = easy_prompt_maker(spider_schema, row['question'], row['db_id'], schema_links)
            model_inputs = tokenizer(prompt, return_tensors='pt')
            model.eval()
            while SQL is None:
                try:
                    with torch.no_grad():
                        tokens = {k:v.to("cuda") for k,v in model_inputs.items()}
                        outputs = model.generate(
                            **tokens,
                            max_new_tokens=64,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=False,
                        )
                        SQL = tokenizer.decode(outputs[0], skip_special_token=True)
                        SQL = SQL.strip()
                except:
                    time.sleep(3)
                    pass
            try:
                SQL = SQL.split("SQL: ")[-1]
                SQL = SQL.split('\n')[0]
                print(SQL)
            except:
                print("SQL slicing error")
                SQL = "SQL"

        elif 'NON-NESTED"' in predicted_class:
            print("NON-NESTED")
            prompt = medium_prompt_maker(spider_schema,spider_foreign, row['question'], row['db_id'], schema_links)
            #print(prompt)
            model_inputs = tokenizer(prompt, return_tensors='pt')
            model.eval()
            while SQL is None:
                try:
                    with torch.no_grad():
                        tokens = {k:v.to("cuda") for k,v in model_inputs.items()}
                        outputs = model.generate(
                            **tokens,
                            max_new_tokens=128,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=False,
                        )
                        SQL = tokenizer.decode(outputs[0], skip_special_token=True)
                        SQL = SQL.strip()
                except:
                    time.sleep(3)
                    pass
            try:
                SQL = SQL.split("SQL: ")[-1]
                #print(SQL)
            except:
                print("SQL slicing error")
                SQL = "SQL"
        else:
            sub_questions = classification.split('questions')[-1]
            sub_questions = sub_questions.split(' = ["')[-1]
            sub_questions = sub_questions.split('"]')[0]
            print(sub_questions)
            print("NESTED")
            prompt = hard_prompt_maker(spider_schema, spider_foreign, row['question'], row['db_id'], schema_links, sub_questions)
            model_inputs = tokenizer(prompt, return_tensors='pt')
            model.eval()
            while SQL is None:
                try:
                    with torch.no_grad():
                        tokens = {k:v.to("cuda") for k,v in model_inputs.items()}
                        outputs = model.generate(
                            **tokens,
                            max_new_tokens=128,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=False,
                        )
                        SQL = tokenizer.decode(outputs[0], skip_special_token=True)
                        SQL = SQL.strip()
                except:
                    timpe.sleep(3)
                    pass
            try:
                SQL = SQL.split("SQL: ")[1]
            except:
                print("SQL slicing error")
                SQL = "SELECT"
        debugged_sql = None
       ## sub_questions = classification.split('questions')[-1]
       # sub_questions = sub_questions.split(' = ["')[-1]
       # sub_questions = sub_questions.split('"]')[0]
        
        prompt = debugger2(spider_schema, spider_foreign, spider_primary, row['question'], row['db_id'], SQL)
        model_inputs = tokenizer(prompt, return_tensors='pt')
        model.eval()
        while debugged_sql is None:
            try:
                with torch.no_grad():
                    tokens = {k:v.to("cuda") for k,v in model_inputs.items()}
                    outputs = model.generate(
                        **tokens,
                        max_new_tokens=128,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=False,
                    )
                    debugged_sql = tokenizer.decode(outputs[0], skip_special_token=True)
                    debugged_sql = debugged.strip()

            except:
                time.sleep(3)
                pass


        SQL = "SELECT" + debugged_sql
        print(SQL)
        CODEX.append([row['question'], SQL, row['query'], row['db_id']])
        df = pd.DataFrame(CODEX, columns=['NLQ', 'PREDICTED SQL', 'GOLD SQL', 'DATABASE'])
        df.to_csv('dsql.csv')

    results = df['PREDICTED SQL'].tolist()
        
    with open('dinsql.csv', 'w') as f:
        for line in results:
            f.write(f"{line}\n")


if __name__ == "__main__":
    fire.Fire(main)
