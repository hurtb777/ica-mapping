import os
from jinja2 import Environment, FileSystemLoader
from io import BytesIO
import matplotlib.pyplot as plt
import base64

PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def create_html(obj_dict, html_filename, template_html_filename='basic_report.html'):
    context = obj_dict
    for fig_name in ['mri_view', 'time_series']:
        plt.figure(obj_dict[fig_name].number)
        # fig = obj_dict[fig_name]  # make it current figure (?hack)
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        figdata_png = base64.b64encode(figfile.getvalue())
        context.update({fig_name: figdata_png.decode('utf8')})

    with open(html_filename, 'w') as f:
        html = render_template(template_html_filename, context)
        f.write(html)
