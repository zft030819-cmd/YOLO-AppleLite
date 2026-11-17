"""Check data recovery safety and exact split/label preservation with a tiny fixture."""
import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
class Restore(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name);self.images=self.base/'original';self.labels=self.base/'labels';self.images.mkdir();self.labels.mkdir();self.rows=[]
        for i,split in enumerate(['train','val','test']):
            rel=f'old/images/{i}.jpg';label=f'old/labels/{i}.txt';b=b'fixture image '+bytes([i]);lb=b'0 0.5 0.5 0.2 0.2\n'
            p=self.images/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);p=self.labels/label;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(lb)
            self.rows.append(dict(image=rel,label=label,split=split,group=str(i),image_sha256=sha(b),label_sha256=sha(lb)))
        self.manifest=self.base/'manifest.json';self.save();self.output=self.base/'restored'
    def save(self):self.manifest.write_text(json.dumps(self.rows))
    def run_restore(self,*extra):
        return subprocess.run([sys.executable,str(ROOT/'scripts/restore_data.py'),'--images-root',str(self.images),'--labels',str(self.labels),'--manifest',str(self.manifest),*extra],capture_output=True,text=True)
    def test_restore_is_exact_and_originals_unchanged(self):
        before={str(p):p.read_bytes() for p in self.images.rglob('*') if p.is_file()};r=self.run_restore('--output',str(self.output));self.assertEqual(r.returncode,0,r.stderr)
        cfg=json.loads((self.output/'data.yaml').read_text());self.assertEqual(len(Path(cfg['test']).read_text().splitlines()),1)
        for row in self.rows:self.assertEqual(sha((self.output/row['label']).read_bytes()),row['label_sha256'])
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.images.rglob('*') if p.is_file()})
    def test_hash_failure_does_not_create_output(self):
        (self.labels/self.rows[0]['label']).write_text('changed');r=self.run_restore('--output',str(self.output));self.assertNotEqual(r.returncode,0);self.assertFalse(self.output.exists())
    def test_group_leakage_is_rejected(self):
        self.rows[1]['group']=self.rows[0]['group'];self.save();self.assertNotEqual(self.run_restore('--check-only').returncode,0)
    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir();(self.output/'keep').write_text('keep');r=self.run_restore('--output',str(self.output));self.assertNotEqual(r.returncode,0);self.assertEqual((self.output/'keep').read_text(),'keep')
    def test_filename_fallback_requires_matching_hash(self):
        src=self.images/self.rows[0]['image'];dst=self.images/'elsewhere'/src.name;dst.parent.mkdir();src.rename(dst);self.assertEqual(self.run_restore('--check-only').returncode,0)
    def test_path_traversal_is_rejected(self):
        self.rows[0]['image']='../outside.jpg';self.save();self.assertNotEqual(self.run_restore('--check-only').returncode,0)
if __name__=='__main__':unittest.main(verbosity=2)
