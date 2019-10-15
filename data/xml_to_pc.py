from bs4 import BeautifulSoup, Tag, NavigableString
from pprint import pprint


"""
xml is too complicated to feed into an LSTM, so I use the function xml_to_pc below to convert xml to a simpler pseudocode.
For example
<note><pitch>C</pitch></note>
would go to
['note', 'pitch', 'C', '}', '}']
The tag name is used to implicity start the tag, and the tag is ended by the character '}'.
"""

def tag_to_pc(tag):
    # converts a tag to pseudocode, like <tagName>x y z</tagName>  -->  ['tagName', 'x', 'y', 'z', '}']
    children = tag.findChildren(recursive=False)
    if not children:
        if tag.string:
            pc = [tag.name]
            for x in tag.string.split():
                pc.append(x)
            pc.append('}')
            return pc
        else:
            return [tag.name, '}']
    else:
        pc = [tag.name]
        for child in children:
            pc.extend(tag_to_pc(child))
        pc.append('}')
        return pc

def clean_pitches(soup):
    """
    Alter the musicxml notation for pitches to use fewer tags
    Thus change <pitch> <step>C</step><alter>0</alter><octave>4</octave></pitch> to <pitch>C 0 4</pitch>
    """
    pitches = soup.find_all('pitch')
    for pitch in pitches:
        step = pitch.find('step')
        alter = pitch.find('alter')
        octave = pitch.find('octave')
        step_str = step.string
        if alter:
            alter_str = alter.string
        else:
            alter_str = str(0)
        octave_str = octave.string
        pitch_str = ' '.join([step_str, alter_str, octave_str])
        new_pitch_element = Tag(name='pitch')
        new_pitch_element.string = pitch_str
        pitch.replace_with(new_pitch_element)



def clean_dynamics(soup):
    """Change the dynamics tag to something like <ff/> of <f/> etc"""
    direction_tags = soup.find_all('direction')
    if direction_tags:
        for direction_tag in direction_tags:
            dynamic_tag = direction_tag.find(
                ['ff', 'f', 'mf', 'mp', 'p', 'pp'])
            direction_tag.replace_with(dynamic_tag)
    return soup


def clean_slurs(soup):
    """Change the slur tag to a simpler <slur/>"""
    notations_tags = soup.find_all('notations')
    if notations_tags:
        for notations_tag in notations_tags:
            notations_tag.replace_with(Tag(name='slur'))
    return soup


def get_key_and_time(soup):
    # get the key signature and time signature information from the musicxml
    # this is not stored in the pseudocode, and must be kept elsewhere
    # key is in range(-7, 8) and measure_length is in 8, 12, or 16.
    key = int(soup.find('fifths').string)
    measure_length = 4*int(soup.find('beats').string)
    return key, measure_length


def xml_to_pc(soup):
    # converts xml to a psuedo code where
    # <tagName>x y z</tagName>  -->  ['tagName', 'x', 'y', 'z', '}']
    clean_pitches(soup)
    clean_dynamics(soup)
    clean_slurs(soup)
    attributes = soup.find('attributes')
    attributes.decompose()
    pc = []
    measures = soup.find_all('measure')
    for measure in measures:
        pc.extend(tag_to_pc(measure))
    return pc


# with open('sample_measure.musicxml') as f:
#     soup = BeautifulSoup(f, 'xml')
#     pc = xml_to_pc(soup)

# print(pc)


# xml = BeautifulSoup('<measure><attributes></attributes><note><pitch><step>G</step></pitch></octave></measure>', 'xml')
# print(xml_to_pc(xml))