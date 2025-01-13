import subprocess
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Generate an SQL dump of the PostgreSQL database associated with this Django project.'

    def handle(self, *args, **options):
        # Get database settings from Django
        db_settings = settings.DATABASES['default']

        # Ensure the database engine is PostgreSQL
        if 'postgresql' not in db_settings['ENGINE']:
            self.stderr.write("This command only supports PostgreSQL databases.")
            return

        # Extract database credentials
        db_name = db_settings['NAME']
        db_user = db_settings['USER']
        db_password = db_settings.get('PASSWORD', '')
        db_host = db_settings.get('HOST', 'localhost')
        db_port = db_settings.get('PORT', '5432')

        # Prompt the user for the output file name
        output_file = f"{db_name}_dump.sql"

        # Construct the pg_dump command
        command = [
            'pg_dump',
            '--dbname', f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
            '-f', output_file
        ]

        try:
            # Execute the pg_dump command
            subprocess.run(command, check=True)
            self.stdout.write(f"SQL dump created successfully: {output_file}")
        except subprocess.CalledProcessError as e:
            self.stderr.write(f"Error occurred while creating the dump: {e}")
        except FileNotFoundError:
            self.stderr.write("pg_dump is not installed or not found in PATH.")
