import pyoverleaf
api = pyoverleaf.Api()
api.login_from_browser()
projects = api.get_projects()
project_id = projects[0].id
rootdir = api.project_get_files(project_id)