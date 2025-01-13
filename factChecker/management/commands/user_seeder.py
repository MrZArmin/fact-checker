from django.core.management.base import BaseCommand
from django.contrib.auth.models import User

class Command(BaseCommand):
    help = 'Seeds a user to the database'

    def add_arguments(self, parser):
        parser.add_argument('username', type=str, help='Username of the user')
        parser.add_argument('password', type=str, help='Password of the user')

    def handle(self, *args, **kwargs):
        username = kwargs['username']
        password = kwargs['password']

        if User.objects.filter(username=username).exists():
            self.stdout.write(self.style.ERROR(
                f'User with username {username} already exists'))
        else:
            User.objects.create_user(
                username=username,
                password=password
            )
            self.stdout.write(self.style.SUCCESS(
                f'Successfully created user with username {username}'))