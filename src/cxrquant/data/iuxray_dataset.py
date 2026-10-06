"""
IU-XRAY (OpenI/NLM) dataset parser and PyTorch DataLoader.
Supports paired image-report loading for CXR report generation.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cxrquant.paths import DATA_DIR, DATA_JSON


class IUXRayParser:
    """Parse IU-XRAY XML reports into structured records."""

    def __init__(self, reports_dir: str, images_dir: str):
        self.reports_dir = Path(reports_dir)
        self.images_dir = Path(images_dir)

    def parse_report(self, xml_path: Path) -> dict | None:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            return None

        uid_elem = root.find('.//uId')
        uid = uid_elem.get('id') if uid_elem is not None else ''

        abstract = {}
        for ab in root.findall('.//AbstractText'):
            label = ab.get('Label', '').upper()
            text = (ab.text or '').strip()
            text = re.sub(r'\s+', ' ', text)
            abstract[label] = text

        mesh_labels = []
        for mesh in root.findall('.//MeSH/major'):
            if mesh.text:
                mesh_labels.append(mesh.text.strip())

        images = []
        for img_elem in root.findall('.//parentImage'):
            img_id = img_elem.get('id', '')
            img_file = self.images_dir / f'{img_id}.png'
            if img_file.exists():
                caption = ''
                cap_elem = img_elem.find('caption')
                if cap_elem is not None and cap_elem.text:
                    caption = cap_elem.text.strip()
                images.append({'id': img_id, 'path': str(img_file), 'caption': caption})

        findings = abstract.get('FINDINGS', '')
        impression = abstract.get('IMPRESSION', '')

        if not findings and not impression:
            return None

        report_text = ''
        if findings:
            report_text += f'FINDINGS: {findings}'
        if impression:
            if report_text:
                report_text += ' '
            report_text += f'IMPRESSION: {impression}'

        return {
            'uid': uid,
            'findings': findings,
            'impression': impression,
            'report': report_text,
            'indication': abstract.get('INDICATION', ''),
            'comparison': abstract.get('COMPARISON', ''),
            'mesh_labels': mesh_labels,
            'images': images,
            'is_normal': 'normal' in [l.lower() for l in mesh_labels],
        }

    def build_dataset(self) -> list[dict]:
        records = []
        xml_files = sorted(self.reports_dir.glob('*.xml'))

        for xml_path in xml_files:
            record = self.parse_report(xml_path)
            if record is None:
                continue
            records.append(record)

        return records

    def build_paired_dataset(self) -> list[dict]:
        """Return only records that have at least one matching PNG image."""
        all_records = self.build_dataset()
        return [r for r in all_records if len(r['images']) > 0]

    def save_to_json(self, output_path: str, paired_only: bool = False):
        records = self.build_paired_dataset() if paired_only else self.build_dataset()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        return records


def create_splits(records: list[dict], train_ratio=0.7, val_ratio=0.1, seed=42):
    """Create train/val/test splits by unique patient UID."""
    rng = np.random.default_rng(seed)
    uids = [r['uid'] for r in records]
    unique_uids = sorted(set(uids))
    rng.shuffle(unique_uids)

    n = len(unique_uids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_uids = set(unique_uids[:n_train])
    val_uids = set(unique_uids[n_train:n_train + n_val])
    test_uids = set(unique_uids[n_train + n_val:])

    splits = {'train': [], 'val': [], 'test': []}
    for r in records:
        uid = r['uid']
        if uid in train_uids:
            splits['train'].append(r)
        elif uid in val_uids:
            splits['val'].append(r)
        elif uid in test_uids:
            splits['test'].append(r)

    return splits


# ── PyTorch Dataset (optional import) ──────────────────────────────────────

try:
    from PIL import Image
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms

    class IUXRayDataset(Dataset):
        def __init__(
            self,
            records: list[dict],
            image_size: int = 224,
            augment: bool = False,
            max_report_len: int = 256,
        ):
            self.records = records
            self.max_report_len = max_report_len

            base_transforms = [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]

            if augment:
                base_transforms = [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ] + base_transforms

            self.transform = transforms.Compose(base_transforms)

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> dict:
            record = self.records[idx]

            images = []
            for img_info in record['images'][:2]:  # max 2 views (PA + lateral)
                img = Image.open(img_info['path']).convert('L')  # grayscale
                img = self.transform(img)
                images.append(img)

            if len(images) == 0:
                images = [torch.zeros(1, 224, 224)]
            if len(images) == 1:
                images.append(images[0].clone())  # duplicate if only 1 view

            return {
                'uid': record['uid'],
                'image_pa': images[0],
                'image_lat': images[1],
                'report': record['report'],
                'findings': record['findings'],
                'impression': record['impression'],
                'is_normal': torch.tensor(record['is_normal'], dtype=torch.float32),
                'mesh_labels': record['mesh_labels'],
            }

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False


if __name__ == '__main__':
    import sys

    reports_dir = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / 'ecgen-radiology')
    images_dir = sys.argv[2] if len(sys.argv) > 2 else str(DATA_DIR)

    parser = IUXRayParser(reports_dir, images_dir)

    print('Parsing all reports...')
    all_records = parser.build_dataset()
    print(f'  Total records: {len(all_records)}')

    paired = parser.build_paired_dataset()
    print(f'  Paired (with images): {len(paired)}')

    splits = create_splits(paired)
    print(f'  Train: {len(splits["train"])}')
    print(f'  Val:   {len(splits["val"])}')
    print(f'  Test:  {len(splits["test"])}')

    # Save structured dataset
    output = str(DATA_JSON)
    parser.save_to_json(output, paired_only=True)
    print(f'Saved to {output}')

    # Sample
    sample = paired[0]
    print('\nSample record:')
    print(f'  UID: {sample["uid"]}')
    print(f'  Images: {len(sample["images"])}')
    print(f'  Findings: {sample["findings"][:100]}...')
    print(f'  MeSH: {sample["mesh_labels"][:3]}')
