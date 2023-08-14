# Setting style for bar graphs

import matplotlib.pyplot as plt

#matplotlib inline

# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'STIXGeneral'

fig, ax = plt.subplots(figsize=(8.5, 11))

# Decorative Lines

ax.axvline(x=.5, ymin=0, ymax=1, color='#007ACC', alpha=0.0, linewidth=50)

plt.axvline(x=.99, color='#000000', alpha=0.5, linewidth=300)

plt.axhline(y=.88, xmin=0, xmax=1, color='#ffffff', linewidth=3)

# set background color

ax.set_facecolor('white')

# remove axes

plt.axis('off')

# add text

plt.annotate('Name', (.02,.94), weight='bold', fontsize=20)

plt.annotate('Education', (.02,.74), weight='bold', fontsize=14)

plt.annotate('Education Description:', (.04,.72), weight='bold', fontsize=10)

plt.annotate('Work History', (.02,.64), weight='bold', fontsize=14)

plt.annotate('Work History Description:', (.04,.62), weight='bold', fontsize=10)

plt.annotate('Certificates', (.02,.54), weight='bold', fontsize=14)

plt.annotate('Certificates Description:', (.04,.52), weight='bold', fontsize=10)

plt.annotate('Security Clearance', (.02,.44), weight='bold', fontsize=14)

plt.annotate('Security Clearance Description:', (.04,.42), weight='bold', fontsize=10)

plt.annotate('Skills', (.02,.98), weight='bold', fontsize=14)

plt.annotate('Skills Description:', (.02,.91), weight='bold', fontsize=10)

plt.annotate('', (.7,.906), weight='regular', fontsize=8, color='#ffffff')

plt.annotate('', (.02,.86), weight='bold', fontsize=10, color='#58C1B2')

plt.annotate('', (.02,.832), weight='bold', fontsize=10)

plt.annotate('', (.04,.78), weight='regular', fontsize=9)

plt.annotate('', (.02,.745), weight='bold', fontsize=10)

plt.annotate('', (.04,.71), weight='regular', fontsize=9)

plt.annotate('', (.02,.672), weight='bold', fontsize=10)

plt.annotate('', (.04,.638), weight='regular', fontsize=9)

plt.annotate('', (.02,.6), weight='bold', fontsize=10)

plt.annotate('', (.02,.54), weight='bold', fontsize=10, color='#58C1B2')

plt.annotate('', (.02,.508), weight='bold', fontsize=10)

plt.annotate('', (.02,.493), weight='regular', fontsize=9, alpha=.6)

plt.annotate('', (.04,.445), weight='regular', fontsize=9)

plt.annotate('', (.02,.4), weight='bold', fontsize=10)

plt.annotate('', (.02,.385), weight='regular', fontsize=9, alpha=.6)

plt.annotate('', (.04,.337), weight='regular', fontsize=9)

plt.annotate('', (.02,.295), weight='bold', fontsize=10)

plt.annotate('', (.02,.28), weight='regular', fontsize=9, alpha=.6)

plt.annotate('', (.04,.247), weight='regular', fontsize=9)

plt.annotate('', (.02,.185), weight='bold', fontsize=10, color='#58C1B2')

plt.annotate('', (.02,.155), weight='bold', fontsize=10)

plt.annotate('', (.02,.14), weight='regular', fontsize=9, alpha=.6)

plt.annotate('', (.04,.125), weight='regular', fontsize=9)

plt.annotate('', (.02,.08), weight='bold', fontsize=10)

plt.annotate('', (.02,.065), weight='regular', fontsize=9, alpha=.6)

plt.annotate('', (.7,.8), weight='bold', fontsize=10, color='#ffffff')

plt.annotate('', (.7,.56), weight='regular', fontsize=10, color='#ffffff')

plt.annotate('', (.7,.43), weight='bold', fontsize=10, color='#ffffff')

plt.annotate('', (.7,.345), weight='regular', fontsize=10, color='#ffffff')

plt.annotate('', (.7,.2), weight='bold', fontsize=10, color='#ffffff')

plt.savefig('resumeexample.png', dpi=300, bbox_inches='tight')