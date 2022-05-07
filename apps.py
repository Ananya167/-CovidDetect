from django.apps import AppConfig
import html
import pathlib
import os


class MlappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'mlapp'
