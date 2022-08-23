import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=False)

# from .draw_utils import COLOR, MARKER_STYLE


COLOR = ((1, 0, 0),
         (0, 1, 0),
         (1, 0, 1),
         (0.5, 0, 0),
         (0, 162 / 255, 232 / 255),
         (0.5, 0.5, 0.5),
         (0, 0, 1),
         (0, 1, 1),
         (136 / 255, 0, 21 / 255),
         (255 / 255, 127 / 255, 39 / 255),
         (0, 0, 0))

LINE_STYLE = ['-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '-']

MARKER_STYLE = ['o', '.', '<', '*', 'D', '.', '.', 'x', '.', '.', 'o']


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=False)


def draw_eao(result):
    fig = plt.figure(figsize=(4, 4),)
    ax = fig.add_subplot(111, projection='polar')

    # fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=True)

    attr2value = []

    # tarcker_names = ['SiamRPN++_Normal', 'SiamRPN++_Normal',

    for i, (tracker_name, ret) in enumerate(result.items()):
        value = list(ret.values())
        print(value, tracker_name)
        attr2value.append(value)
        value.append(value[0])

    attr2value = np.array(attr2value)
    max_value = np.max(attr2value, axis=0)
    min_value = np.min(attr2value, axis=0)
    for i, (tracker_name, ret) in enumerate(result.items()):
        value = list(ret.values())
        value.append(value[0])
        value = np.array(value)
        value *= (1 / max_value)
        plt.plot(angles, value, linestyle='-', color=COLOR[i], marker=MARKER_STYLE[i],
                 label=tracker_name, linewidth=1.5, markersize=6)

    attrs = ["Overall", "Camera motion",
             "Illumination change", "Motion Change",
             "Size change", "Occlusion",
             "Unassigned"]
    attr_value = []
    for attr, maxv, minv in zip(attrs, max_value, min_value):
        attr_value.append(attr + "\n({:.3f},{:.3f})".format(minv, maxv))
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, attr_value)
    ax.spines['polar'].set_visible(False)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=3)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 3, 4, 5, 2, 1]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
              loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=3)

    ax.grid(b=False)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([])
    plt.show()
    # plt.savefig('img.png')
    fig.savefig('vot_attributes.png', dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    result = pickle.load(open("../../tools/VOT2018.pickle", 'rb'))
    draw_eao(result)
