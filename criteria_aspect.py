class CriteriaAspect:
    def __init__(self, number: int, aspect: str, definition: str):
        self.number = number
        self.aspect = aspect
        self.definition = definition
        self.questions = []
    
    def __str__(self):
        return f"{self.number}. {self.aspect}: {self.definition}"
    
    def add_question(self, question: str) -> None:
        """Adds a criteria question in a formatted string to the questions list."""
        question_str = f"\t{self.number}.{len(self.questions)+1}: {question}"
        self.questions.append(question_str)

    def get_all_questions(self) -> str:
        """Returns a string of questions."""
        return ("\n").join(self.questions)