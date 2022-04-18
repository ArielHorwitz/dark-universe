# dark universe main running module

from nutil import *
import gui
import universe

ALLOW_PROMPT = False

save_path = universe.SAVE_DIR
save_files = [*save_path.glob('*.dus')]
save_files_display = [f'* {sf}' for sf in save_files]

if __name__ == '__main__':
    print(make_title(f'Starting {universe.FULL_TITLE}'))
    if ALLOW_PROMPT:
        if len(save_files) > 0:
            while True:
                print('\nFound save files:')
                print(adis(save_files_display, split_threshold=0))
                print('\nLeave blank for new game, enter "q" to quit.')
                loadprompt = input('Load save >> ')
                if loadprompt == 'q':
                    quit()
                elif loadprompt != '':
                    try:
                        uni = universe.Universe.do_import(savename=loadprompt)
                    except (FileNotFoundError) as e:
                        print(sjoin([
                            '\n\n',
                            '-'*20,
                            'No such save found...',
                            ]))
                        continue
                    gui.App(uni=uni).start()
                    quit()
                else:
                    print('Starting new game.')
                    break
        else:
            print('No saves found, starting new game.')
    # tk gui
    # gui_tk.App().start()
    # kivy gui
    gui.App().run()
