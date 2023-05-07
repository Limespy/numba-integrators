import datetime
import pathlib
import re

import numba_integrators as ni
import yamdog as md

PATH_BASE = pathlib.Path(__file__).parent
PATH_README = PATH_BASE / 'README.md'
PATH_CHANGELOG = PATH_BASE / '.changelog.md'
PATH_PYPROJECT = PATH_BASE / 'pyproject.toml'
VERSION = ni.__version__

re_heading = re.compile(r'^#* .*$')

def parse_md_element(text: str):
    if match := re_heading.match(text):
        hashes, content = match[0].split(' ', 1)
        return md.Heading(content, len(hashes))
    else:
        return md.Raw(text)
#-----------------------------------------------------------------------
def parse_md(text: str):
    return md.Document([parse_md_element(item.strip())
                        for item in text.split('\n\n')])
#-----------------------------------------------------------------------
def make_changelog(level: int):
    doc = md.Document([md.Heading('Changelog', level, in_TOC = False)])
    changelog = parse_md(PATH_CHANGELOG.read_text())
    if changelog:
        if (latest := changelog.content[0]).content.split(' ', 1)[0] == VERSION:
            latest.content = f'{VERSION} {datetime.date.today().isoformat()}'
        else:
            raise ValueError('Changelog not up to date')

        PATH_CHANGELOG.write_text(str(changelog) + '\n')

        for item in changelog:
            if isinstance(item, md.Heading):
                item.level = level + 1
                item.in_TOC = False

        doc += changelog

    return doc
#=======================================================================
def make(name, pypiname, description):

    shields_url = 'https://img.shields.io/'

    pypi_project_url = f'https://pypi.org/project/{pypiname}'
    pypi_badge_info = (('v', 'PyPI Package latest release'),
                       ('wheel', 'PyPI Wheel'),
                       ('pyversions', 'Supported versions'),
                       ('implementation', 'Supported implementations'))
    pypi_badges = [md.Link(md.Image(f'{shields_url}pypi/{code}/{pypiname}.svg',
                                    desc),
                            pypi_project_url,
                           '')
                   for code, desc in pypi_badge_info]

    doc = md.Document([md.Heading(f'Overview of {name}', 1),
                       md.Paragraph(pypi_badges, '\n'),
                       description])
    doc += make_changelog(1)
    return doc
#=======================================================================
def main():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    pyproject = tomllib.loads(PATH_PYPROJECT.read_text())
    master_info = pyproject['master-info']
    package_name = master_info["package_name"]
    full_name = master_info.get("full_name",
                                package_name.replace('-', ' ').capitalize())
    description = master_info['description']

    doc = make(full_name, package_name, description)
    PATH_README.write_text(str(doc) + '\n')
#=======================================================================
if __name__ == '__main__':
    main()
