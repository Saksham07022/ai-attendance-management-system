import os
import json
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ---------------- CONFIGURATION ----------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "students.json"

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
MODEL = os.getenv("HF_MODEL") or os.getenv("MODEL") or "Qwen/Qwen2.5-7B-Instruct"

START_DATE = date(2026, 10, 8)
END_DATE = date(2026, 10, 14)

DATES = [
    (START_DATE + timedelta(days=i)).isoformat()
    for i in range((END_DATE - START_DATE).days + 1)
]

client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None


# ---------------- JSON FILE ----------------

def create_sample_data():
    """Create sample data if students.json is missing or empty."""
    students = [
        {"name": "Aarav Sharma", "roll_no": "101"},
        {"name": "Priya Singh", "roll_no": "102"},
        {"name": "Rahul Verma", "roll_no": "103"},
        {"name": "Ananya Gupta", "roll_no": "104"},
        {"name": "Arjun Kumar", "roll_no": "105"},
    ]

    for student in students:
        student["attendance"] = {}

        # Sample attendance for demonstration
        for i, day in enumerate(DATES):
            student["attendance"][day] = (
                i < 5 if student["roll_no"] == "101"
                else i < 4 if student["roll_no"] == "102"
                else i < 6 if student["roll_no"] == "103"
                else i < 3 if student["roll_no"] == "104"
                else i < 5
            )

    return {"students": students}


def save_data(data):
    """Save updated attendance to JSON."""
    temp_file = DATA_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    temp_file.replace(DATA_FILE)


def load_data():
    """Load data or create a valid sample file."""
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:
        data = create_sample_data()
        save_data(data)
        return data

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(
        data.get("students"), list
    ):
        raise ValueError(
            "students.json must contain a students list."
        )

    return data


# ---------------- ATTENDANCE CALCULATIONS ----------------

def get_student(data, roll_no):
    for student in data["students"]:
        if str(student["roll_no"]) == str(roll_no):
            return student

    raise ValueError(f"Roll number {roll_no} not found.")


def attendance_report(student):
    attendance = student["attendance"]

    present_days = sum(
        attendance.get(day) is True for day in DATES
    )

    total_days = len(DATES)
    percentage = round((present_days / total_days) * 100, 2)

    return {
        "name": student["name"],
        "roll_no": str(student["roll_no"]),
        "present_days": present_days,
        "total_days": total_days,
        "attendance_percentage": percentage
    }


def class_report(data, selected_date):
    reports = [
        attendance_report(student)
        for student in data["students"]
    ]

    average = round(
        sum(r["attendance_percentage"] for r in reports)
        / len(reports), 2
    ) if reports else 0

    present_today = sum(
        student["attendance"].get(selected_date) is True
        for student in data["students"]
    )

    return {
        "status": "success",
        "date": selected_date,
        "class_size": len(reports),
        "present_today": present_today,
        "absent_or_unmarked_today": len(reports) - present_today,
        "class_average_attendance_percentage": average,
        "students": reports
    }


# ---------------- HUGGING FACE AI ----------------

def understand_request(user_request, data):
    if client is None:
        raise ValueError(
            "HF_TOKEN missing. Add your Hugging Face token to .env."
        )

    student_list = [
        {
            "name": s["name"],
            "roll_no": str(s["roll_no"])
        }
        for s in data["students"]
    ]

    prompt = f"""
You are an intent parser for a student attendance management system.

Available students: {json.dumps(student_list)}
Allowed dates: 2026-10-08 through 2026-10-14.

Convert the user's request into ONE JSON object only.
Never invent student details or dates.

Allowed actions:

1. Mark attendance:
{{"action":"mark","roll_no":"101","date":"2026-10-10","present":true}}

2. Remove an attendance record:
{{"action":"remove","roll_no":"101","date":"2026-10-10"}}

3. Student attendance report:
{{"action":"report","roll_no":"101","date":"2026-10-10"}}

4. Class report:
{{"action":"class","date":"2026-10-10"}}

5. Clarification:
{{"action":"clarify","message":"Ask the user for the missing information"}}

For mark, present must be true or false.
Use dates in YYYY-MM-DD format.
Convert dates such as "10 October 2026" to "2026-10-10".
If information is missing or ambiguous, return clarify.
Return valid JSON only.

User request: {user_request}
"""

    response = client.chat_completion(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Convert requests to valid JSON actions only."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise ValueError("Hugging Face returned an empty response.")

    content = content.strip()
    content = content.removeprefix("```json").removesuffix("```").strip()

    action = json.loads(content)

    if not isinstance(action, dict):
        raise ValueError("AI did not return a JSON object.")

    return action


# ---------------- EXECUTE AI ACTION ----------------

def validate_date(value):
    try:
        selected = date.fromisoformat(value)
    except (ValueError, TypeError):
        raise ValueError("Invalid date. Use 8-14 October 2026.")

    if selected < START_DATE or selected > END_DATE:
        raise ValueError("Date must be between 8 and 14 October 2026.")

    return selected.isoformat()


def execute_action(action, data):
    kind = action.get("action")

    if kind == "clarify":
        return {
            "status": "clarification_needed",
            "message": action.get("message", "Please clarify your request.")
        }

    if kind not in ("mark", "remove", "report", "class"):
        raise ValueError("Unsupported AI action.")

    selected_date = validate_date(action.get("date", ""))

    if kind == "class":
        return class_report(data, selected_date)

    if "roll_no" not in action:
        raise ValueError("Roll number is required.")

    student = get_student(data, action["roll_no"])

    if kind == "mark":
        present = action.get("present")

        if not isinstance(present, bool):
            raise ValueError("Attendance must be true or false.")

        student["attendance"][selected_date] = present
        save_data(data)

        result = attendance_report(student)
        result.update({
            "status": "success",
            "action": "mark",
            "date": selected_date,
            "present": present,
            "message": "Attendance updated and saved to students.json."
        })
        return result

    if kind == "remove":
        existed = selected_date in student["attendance"]
        student["attendance"].pop(selected_date, None)
        save_data(data)

        return {
            "status": "success",
            "action": "remove",
            "name": student["name"],
            "roll_no": str(student["roll_no"]),
            "date": selected_date,
            "record_removed": existed,
            "message": "Attendance record removed from students.json."
        }

    if kind == "report":
        result = attendance_report(student)
        result.update({
            "status": "success",
            "date": selected_date,
            "attendance_on_date": (
                "unmarked"
                if selected_date not in student["attendance"]
                else "present"
                if student["attendance"][selected_date]
                else "absent"
            )
        })
        return result


# ---------------- TERMINAL INTERFACE ----------------

def main():
    print("\n=== AI ATTENDANCE MANAGEMENT SYSTEM ===")
    print("Dates: 8-14 October 2026")
    print("\nExamples:")
    print("Mark roll 101 present on 10 October 2026")
    print("Mark roll 102 absent on 12 October 2026")
    print("Remove roll 102 attendance on 12 October 2026")
    print("Show roll 103 attendance percentage on 14 October 2026")
    print("Calculate class average on 14 October 2026")
    print("\nType exit to quit.")

    try:
        data = load_data()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({
            "status": "error",
            "message": f"Could not load student data: {exc}"
        }, indent=4))
        return

    while True:
        try:
            user_request = input("\nYou: ").strip()

            if user_request.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if not user_request:
                continue

            action = understand_request(user_request, data)
            result = execute_action(action, data)

            print(json.dumps(result, indent=4, ensure_ascii=False))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as exc:
            print(json.dumps({
                "status": "error",
                "message": str(exc)
            }, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()