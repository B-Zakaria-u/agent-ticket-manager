"""Default mock data for agent testing endpoints."""

DEFAULT_SPEC_TICKET = """
## Bug: `ModuleNotFoundError: No module named 'apps.services.urls'` on project launch

## Description

A `ModuleNotFoundError` is raised when attempting to launch the project, preventing the development server from starting.

## Error

```
ModuleNotFoundError: No module named 'apps.services.urls'
```

## Steps to Reproduce

1. Clone the repository
2. Run the development server:
   ```bash
   python manage.py runserver
   ```
3. Observe the error in the terminal output

## Expected Behavior

The development server starts successfully without any import errors.

## Actual Behavior

The server fails to start due to a missing `urls.py` module inside `apps/services/`.

## Root Cause (Suspected)

The `apps/services/` application is referenced in the project's URL configuration (likely in `fiber_crm/urls.py`), but the `urls.py` file is either missing or has not been created inside the `apps/services/` directory.

## Project Structure

```
fiber-crm/
├── apps/
│   ├── core/
│   ├── leads/
│   ├── customers/
│   ├── services/           # Missing urls.py here
│   └── tickets/
├── fiber_crm/              # urls.py likely references apps.services.urls
├── templates/
├── static/
├── db.sqlite3
├── manage.py
└── seed_data.py
```

## Suggested Fix

Create a `urls.py` file inside `apps/services/`:

```python
# apps/services/urls.py
from django.urls import path
from . import views

app_name = "services"

urlpatterns = [
    # Define service-related routes here
]
```

Then verify that `fiber_crm/urls.py` includes it correctly:

```python
path("services/", include("apps.services.urls")),
```

## Environment

| Key | Value |
|---|---|
| Framework | Django |
| Entry Point | `manage.py runserver` |
| Database | SQLite (`db.sqlite3`) |

## Acceptance criteria
The project should run
"""

DEFAULT_CODING_SPEC = """
  Ticket  : Error when launching the project
  Statut  : CONFIANT (confiance 90%)
 
  --- Localisation du bug ---
  Fichier   : apps/services/urls.py
  Fonction  : include('apps.services.urls')
  Ligne     : 1
  Langage   : python
 
  --- Cause probable ---
  The 'apps.services.urls' module does not exist; the development server cannot start without it.
 
  --- Resumé du probleme ---
  Observed behaviour: The development server fails to start due to a missing 'urls.py' module inside 'apps/services/'. Expected behaviour: The development server starts successfully without any import errors. Trigger condition: The 'apps.services.urls' module is referenced in the project's URL configuration.
 
  --- Contraintes du patch ---
  Scope  : Modifier uniquement include('apps.services.urls')() dans apps/services/urls.py
  Style  : conventions existantes
 
  --- Comportement attendu apres fix ---
  Corriger le bug décrit dans : The 'apps.services.urls' module does not exist; the development server cannot start without it.
 
  --- Contexte du code ---
  L3: import os
  L4: import sys
  L5:
  L6:
  L7: def main():
  L9:     os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fiber_crm.settings')
  L10:     try:
  L11:         from django.core.management import execute_from_command_line
  L12:     except ImportError as exc:
  L13:         raise ImportError(
  L14:             "Couldn't import Django. Are you sure it's installed and "
  L15:             "available on your PYTHONPATH environment variable? Did you "
  L16:             "forget to activate a virtual environment?"
 
"""

DEFAULT_CODING_FILES = {
  "apps/services/urls.py": """from django.urls import path
from . import views

app_name = "services"

urlpatterns = [
    # Define service-related routes here
]""",
}

DEFAULT_TESTING_MOCK = {
  "apps/services/urls.py": """from django.urls import path
from . import views

app_name = "services"

urlpatterns = [
    # Define service-related routes here
]""",
}

DEFAULT_SCOUT_MOCK = {
  "apps/services/urls.py": """from django.urls import path
from . import views

app_name = "services"

urlpatterns = [
    # Define service-related routes here
]""",
}
