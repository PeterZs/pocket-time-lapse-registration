from PIL import Image as PIM
from PIL import ImageDraw as _ImageDraw
from PIL import ImageFont as _ImageFont
import numpy as np
import os
from ptlreg.apy.utils.util_funcs import __current_running_python_version__

def _find_all_files_with_name_under_path(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


###########################adapted from https://gist.github.com/turicas/1455973##########################
def _get_font_size(text, font_path, max_width=None, max_height=None):
    if max_width is None and max_height is None:
        raise ValueError('You need to pass max_width or max_height')
    font_size = 1
    text_size = _get_text_size(font_path, font_size, text)
    if (max_width is not None and text_size[0] > max_width) or (max_height is not None and text_size[1] > max_height):
        raise ValueError("Text can't be filled in only (%dpx, %dpx)" % text_size)
    while True:
        if (max_width is not None and text_size[0] >= max_width) or (max_height is not None and text_size[1] >= max_height):
            return font_size - 1
        font_size += 1
        text_size = _get_text_size(font_path, font_size, text)


def _get_text_size(font_path, font_size, text):
    # print("font path: {}\nfont size:{}\ntext:{}".format(font_path, font_size, text));
    font = _ImageFont.truetype(font_path, font_size)
    return font.getsize(text)







class _ImageText(object):
    @staticmethod
    def text_wrap(text, font, max_width):
        lines = []
        # If the width of the text is smaller than image width
        # we don't need to split it, just add it to the lines array
        # and return
        if font.getsize(text)[0] <= max_width:
            lines.append(text)
        else:
            # split the line by spaces to get words
            words = text.split(' ')
            i = 0
            # append every word to a line while its width is shorter than image width
            while i < len(words):
                line = ''
                while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                    line = line + words[i] + " "
                    i += 1
                if not line:
                    line = words[i]
                    i += 1
                # when the line gets longer than the max width do not append the word,
                # add the line to the lines array
                lines.append(line)
        return lines

    # def withText(self, text=None, xy=None,
    #              alpha=1,
    #              font_size='fill',
    #              font_filename='RobotoCondensed-Regular.ttf',
    #              max_width=None,
    #              max_height=None,
    #              color = (255,255,255),
    #              shadow_color = (0,0,0),
    #              shadow_offset_xy=(3,3),
    #              encoding='utf8', draw_context = None):
    #     imcopy=self.clone(share_data=False);
    #     _ImageText.writeShadowedText(img=imcopy,
    #                                  xy=xy,
    #                                  text=text,
    #                                  font_filename=font_filename,
    #                                  max_width=max_width,
    #                                  max_height=max_height,
    #                                  color=color,
    #                                  shadow_color=shadow_color,
    #                                  shadow_offset_xy=shadow_offset_xy);


    @staticmethod
    def write_shadowed_text(img, text, xy,
                            font_size=None,
                            font_filename=None,
                            max_width=None,
                            max_height=None,
                            color = None,
                            shadow_color = None,
                            shadow_offset_xy=None,
                            encoding='utf8', draw_context = None):
        print(text);
        print(xy);
        if(color is None):
            color = (255, 255, 255);
        if(shadow_color is None):
            shadow_color = (0, 0, 0);
        if(font_size is None):
            font_size = 'fill';
        if(font_filename is None):
            font_filename = 'RobotoCondensed-Regular.ttf';
        if(shadow_offset_xy is None):
            shadow_offset_xy = [3,3];
        _ImageText.write_text(img, xy=[xy[0] + shadow_offset_xy[0], xy[1] + shadow_offset_xy[1]], text=text, font_filename=font_filename,
                              font_size=font_size,
                              max_width=max_width,
                              color=shadow_color,
                              max_height=max_height,
                              encoding=encoding, draw_context=draw_context);
        _ImageText.write_text(img, xy=xy, text=text, font_filename=font_filename,
                              font_size=font_size,
                              max_width=max_width,
                              color=color,
                              max_height=max_height,
                              encoding=encoding, draw_context=draw_context);

    @staticmethod
    def write_text(img, xy, text, font_filename='RobotoCondensed-Regular.ttf',
                   font_size='fill',
                   color=(0, 0, 0),
                   max_width=None,
                   max_height=None,
                   encoding='utf8', draw_context = None):

        x=xy[0];
        y=xy[1];

        font_paths = _find_all_files_with_name_under_path(name=font_filename,
                                                          path=os.path.dirname(os.path.abspath(__file__)));

        font_path = font_paths[0];

        if(__current_running_python_version__()<3):
            if isinstance(text, str):
                text = text.decode(encoding)


        if font_size == 'fill' and (max_width is not None or max_height is not None):
            font_size = _get_font_size(text, font_path, max_width,
                                       max_height)
        text_size = _get_text_size(font_path, font_size, text)


        font = _ImageFont.truetype(font=font_path, size=font_size);

        # font = _ImageFont.truetype(font_filename, font_size)
        if x == 'center':
            x = (img.shape[1] - text_size[0]) / 2
        if y == 'center':
            y = (img.shape[0] - text_size[1]) / 2

        if(draw_context is None):
            ipil = img.pil();
            draw = _ImageDraw.Draw(ipil)
            draw.text((x, y), text, font=font, fill=color)
            datashape = img.pixels.shape;
            img.pixels = np.array(ipil.getdata());
            img.pixels.shape = datashape;
        else:
            draw_context.text((x, y), text, font=font, fill=color);
        return text_size


    @staticmethod
    def write_text_box(img, xy, text, box_width, font_filename='RobotoCondensed-Regular.ttf',
                       font_size=None, color=(0, 0, 0), place='left',
                       justify_last_line=False):
        if(font_size is None):
            font_size = 11;

        x = xy[0];
        y = xy[1];

        font_paths = _find_all_files_with_name_under_path(name=font_filename,
                                                          path=os.path.dirname(os.path.abspath(__file__)));
        font_path = font_paths[0];

        lines = []
        line = []
        words = text.split()
        for word in words:
            new_line = ' '.join(line + [word])
            size = _get_text_size(font_path, font_size, new_line)
            text_height = size[1]
            if size[0] <= box_width:
                line.append(word)
            else:
                lines.append(line)
                line = [word]
        if line:
            lines.append(line)
        lines = [' '.join(line) for line in lines if line]
        height = y
        for index, line in enumerate(lines):
            height += text_height
            if place == 'left':
                _ImageText.write_text(img, (x, height), line,
                                      font_filename,
                                      font_size,
                                      color)
            elif place == 'right':
                total_size = _get_text_size(font_path, font_size, line)
                x_left = x + box_width - total_size[0]
                _ImageText.write_text(img, (x_left, height), line, font_filename,
                                      font_size, color)
            elif place == 'center':
                total_size = _get_text_size(font_path, font_size, line)
                x_left = int(x + ((box_width - total_size[0]) / 2))
                _ImageText.write_text(img, (x_left, height), line,
                                      font_filename,
                                      font_size, color)
            elif place == 'justify':
                words = line.split()
                if (index == len(lines) - 1 and not justify_last_line) or len(words) == 1:
                    _ImageText.write_text(img, (x, height), line,
                                          font_filename,
                                          font_size,
                                          color)
                    continue
                line_without_spaces = ''.join(words)
                total_size = _get_text_size(font_path, font_size,
                                            line_without_spaces)
                space_width = (box_width - total_size[0]) / (len(words) - 1.0)
                start_x = x
                for word in words[:-1]:
                    _ImageText.write_text(img, (start_x, height), word,
                                          font_filename,
                                          font_size, color)
                    word_size = _get_text_size(img, font_path, font_size,
                                               word)
                    start_x += word_size[0] + space_width
                last_word_size = _get_text_size(img, font_path, font_size,
                                                words[-1])
                last_word_x = x + box_width - last_word_size[0]
                _ImageText.write_text(img, (last_word_x, height), words[-1],
                                      font_filename,
                                      font_size, color)
        return (box_width, height - y)


ImageText = _ImageText;

