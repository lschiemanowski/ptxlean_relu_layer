#!/usr/bin/env python3
"""Apply the source compatibility fixes required by our Lean 4.35 pins.

Run after `lake update`, before `lake build`. No arithmetic definitions change.
Only isolated package checkouts inside this project may be patched.
"""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / '.lake' / 'packages'
FIXES = [
    ('floatlib', 'c9a051f08f948ce473036b815b0669fafb61f601',
     'FloatLib/Floats/Formats/BinaryInterchange/Model/Lean.lean', [
         ('((Nat.mod_lt n (Nat.two_pow_pos width)).trans_le hpow)',
          '(Nat.lt_of_lt_of_le (Nat.mod_lt n (Nat.two_pow_pos width)) hpow)')]),
    ('floatlib', 'c9a051f08f948ce473036b815b0669fafb61f601',
     'FloatLib/Floats/Formats/BinaryInterchange/Model/Fields/Optimized.lean', [
         ('hShifted.trans_le', 'Nat.lt_of_lt_of_le hShifted'),
         ('hFractionField.trans_le', 'Nat.lt_of_lt_of_le hFractionField')]),
    ('TorchLean', 'de3192df4779877ceb65cfe470a4d4ce9d480a5b',
     'NN/Tensor/Internal/Representation/Storage.lean', [
         ('decreasing_by\n  have hIndexNat : index.toNat < bound.toNat :=',
          'decreasing_by\n  have _hIndexNat : index.toNat < bound.toNat :=')]),
    ('TorchLean', 'de3192df4779877ceb65cfe470a4d4ce9d480a5b',
     'NN/Spec/Layers/Gru.lean', [
         ('let (hidden, reset, update, candidate, resetHidden) :=',
          'let (hiddenState, reset, update, candidate, resetHidden) :='),
         ('(hidden, (hidden, reset, update, candidate, resetHidden))',
          '(hiddenState, (hiddenState, reset, update, candidate, resetHidden))'),
         ('let (hidden, _, _, _, _) := saved.getScalar i\n    hidden',
          'let (hiddenState, _, _, _, _) := saved.getScalar i\n    hiddenState')]),
]


def git(package, *args):
    return subprocess.check_output(['git', '-C', str(package), *args], text=True).strip()


def main():
    for name, revision, relative, replacements in FIXES:
        package = PACKAGES / name
        if not package.resolve().is_relative_to(PACKAGES):
            raise RuntimeError(f'{name}: use an isolated local checkout, not a shared symlink')
        if git(package, 'rev-parse', 'HEAD') != revision:
            raise RuntimeError(f'{name}: unexpected revision; review compatibility fixes first')
        original = subprocess.check_output(
            ['git', '-C', str(package), 'show', f'{revision}:{relative}'], text=True)
        patched = original
        for old, new in replacements:
            if old not in patched:
                raise RuntimeError(f'{relative}: expected source passage is absent')
            patched = patched.replace(old, new)
        target = package / relative
        current = target.read_text()
        if current == original:
            target.write_text(patched)
            print(f'Applied compatibility fix: {name}/{relative}')
        elif current == patched:
            print(f'Compatibility fix already applied: {name}/{relative}')
        else:
            raise RuntimeError(f'{relative}: unexpected edits; refusing to overwrite them')


if __name__ == '__main__':
    main()
