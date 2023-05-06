import pathlib

import yamdog as md

PATH_README = pathlib.Path(__file__).parent / 'README.md'

def main():
    pypiname = 'numba-integrators'
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

    doc = md.Document([md.Heading(1, 'Numba Integrators'),
                       md.Paragraph(pypi_badges, '\n'),
                       'Numerical integrators using Numba'])
    PATH_README.write_text(str(doc) + '\n')

if __name__ == '__main__':
    main()
