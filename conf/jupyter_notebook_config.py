# Configuration file for jupyter-notebook.

#------------------------------------------------------------------------------
# NotebookApp(JupyterApp) configuration
#------------------------------------------------------------------------------

## Set the Access-Control-Allow-Origin header
#
#  Use '*' to allow any origin to access your server.
#
#  Takes precedence over allow_origin_pat.
c.NotebookApp.allow_origin = '*'

## The IP address the notebook server will listen on.
c.NotebookApp.ip = '0.0.0.0'  #  Should be the same as '*'.

## Whether to open in a browser after starting. The specific browser used is
#  platform dependent and determined by the python standard library `webbrowser`
#  module, unless it is overridden using the --browser (NotebookApp.browser)
#  configuration option.
c.NotebookApp.open_browser = False

## The port the notebook server will listen on.
c.NotebookApp.port = 8888

#------------------------------------------------------------------------------
# InlineBackend configuration
#------------------------------------------------------------------------------

# An object to store configuration of the inline backend.

# The image format for figures with the inline backend.
c.InlineBackend.figure_format = 'retina'


def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    import os
    from subprocess import check_call

    if model['type'] != 'notebook':
        return # only do this for notebooks

    import nbformat
    notebook = nbformat.read(os_path, as_version=nbformat.NO_CONVERT, )

    import nbconvert

    python_exporter = nbconvert.exporters.PythonExporter()
    (body, resources) = python_exporter.from_notebook_node(notebook)

    lines = body.split('\n')
    lines = [x for x in lines if x]  # Remove empty lines.
    lines = [x for x in lines if not x.startswith('get_ipython().magic')]  # Remove magic lines.

    body = '\n'.join(lines) + '\n'

    python_filename = os.path.splitext(os_path)[0] + '.py'
    with open(python_filename, 'w') as py_file:
        try:
            # Python 3
            py_file.write(body)
        except UnicodeEncodeError:
            # Python 2
            py_file.write(body.encode('utf-8'))

    print('Converting notebook {0} to python file {1}...'.format(
          os.path.basename(os_path),
          os.path.basename(python_filename),
          ))


c.FileContentsManager.post_save_hook = post_save
