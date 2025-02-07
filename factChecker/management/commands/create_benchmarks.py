from django.core.management.base import BaseCommand
from factChecker.models import Article, Benchmark
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import os
import json

class Command(BaseCommand):
    help = 'Creates test database after clearing existing data.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_dotenv()
        self.prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.system_prompt = self._load_prompt("create_questions_prompt.txt")
        self.temperature = 0.7

    def handle(self, *args, **kwargs):
        random_articles = Article.objects.order_by('?')[:100]
        self._create_test(random_articles)
        self.stdout.write(self.style.SUCCESS('Successfully created new benchmark data'))

    def _create_test(self, articles):
        for article in articles:
            text = article.title + " " + article.lead + " " + article.text
            answer_str = self._create_question_for_article(text)
            try:
                answer_dict = json.loads(answer_str)
                for answer in answer_dict:
                    Benchmark.objects.create(
                        article=article,
                        question=answer['question'],
                        answer=answer['answer']
                    )
            except json.JSONDecodeError as e:
                self.stdout.write(self.style.ERROR(f'Error parsing JSON for article {str(e)}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error creating benchmark for article {str(e)}'))

    def _create_question_for_article(self,article_text):
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Here is the article: {article_text}"
                }
            ]
            completion_params = {
                "messages": messages,
                "model": self.model,
                "temperature": self.temperature,
            }
            chat_completion = self.client.chat.completions.create(**completion_params)
            return chat_completion.choices[0].message.content
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            raise RuntimeError(error_msg)

    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        try:
            prompt_path = self.prompts_dir / filename
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            raise FileNotFoundError(f"Could not load prompt {filename}: {str(e)}")
