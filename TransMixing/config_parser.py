import json
from pathlib import Path, PurePosixPath
from collections import OrderedDict

class ConfigParser:
    def __init__(self, cfg_fname):
        self.cfg_fname = Path(cfg_fname)
        self.config = self.read_json()

    def read_json(self):
        with self.cfg_fname.open('rt') as handle:
            return json.load(handle, object_hook=OrderedDict)

    def write_json(self, content, cfg_fname_out):
        with cfg_fname_out.open('wt') as handle:
            json.dump(content, handle, indent=4, sort_keys=False)

    def __getitem__(self, key):
        return self.config[key]

if __name__ == '__main__':
    config = ConfigParser('./config.json')
    print(0)
