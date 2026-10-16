import json
import unittest
import numpy as np
import study


class VersionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(610152)
        self.x=(rng.normal(size=(12,4,25))*np.array([1,.2,2,.1])[None,:,None]*1e-5).astype(np.float32)
        self.ids=np.repeat([1,2,3],4)
        self.op=study.load_module(study.SWEEP/'study.py','test_floor_operator')

    def test_design_all_folds(self):
        cfg=json.loads((study.HOME/'config.json').read_text(encoding='utf-8'))
        cells=study.cells(cfg)
        self.assertEqual(len(cells),8)
        self.assertEqual(len({r['id'] for r in cells}),8)
        self.assertEqual(sum(c['version']=='adopted' for c in cells),4)
        self.assertEqual(len(study.NAMES)*len(cells),32)
        self.assertFalse(cfg['target_label_parameter_selection'])

    def test_relative_invariance_both_pipelines(self):
        for pipe in ['subject','dataset_subject']:
            a,_=self.op.align(self.x,self.ids,'relative',pipe,1e-8,1e-8)
            b,_=self.op.align(self.x*np.float32(32),self.ids,'relative',pipe,1e-8,1e-8)
            self.assertLess(study.norm_error(a,b),1e-5)

    def test_unchanged_absolute_is_not_unit_conversion(self):
        a,_=self.op.align(self.x,self.ids,'absolute','subject',1e-8,1e-8)
        b,_=self.op.align(self.x*np.float32(32),self.ids,'absolute','subject',1e-8,1e-8)
        self.assertGreater(study.norm_error(a,b),.01)

    def test_exact_binary_scaling(self):
        a=self.x*np.float32(.03125)
        np.testing.assert_array_equal(a*np.float32(32),self.x)
        self.assertEqual(study.norm_error(a,self.x*np.float32(.03125)),0)

    def test_metadata_guard(self):
        a=dict(X=self.x,y=np.tile([0,1],6),subjects=self.ids,channels=['a','b','c','d'],sfreq=100)
        study.check_metadata(a,dict(a))
        b=dict(a,y=1-a['y'])
        with self.assertRaises(AssertionError): study.check_metadata(a,b)

    def test_archived_operators_match(self):
        old=study.load_module(study.ARCHIVE/'precision_compare.py','test_precision')
        for policy in ['absolute','relative']:
            for pipe in ['subject','dataset_subject']:
                a,_=self.op.align(self.x,self.ids,policy,pipe,1e-8,1e-8)
                b,_=old.align(self.x,self.ids,policy,pipe,1.)
                np.testing.assert_array_equal(a,b)

    def test_non_spd_fails(self):
        with self.assertRaises(ValueError): self.op.batch(np.zeros_like(self.x),'absolute',1e-8)

    def test_K_sign_and_version_pairing(self):
        from analyze import coefficients
        ids=[v+'_'+p+'_'+s for v in ['adopted','original'] for p in ['absolute','relative'] for s in ['subject','dataset_subject']]
        values=np.array([65.,70.,70.,70.2,68.,70.,70.,70.2])
        a=coefficients(ids,'adopted')
        o=coefficients(ids,'original')
        self.assertAlmostEqual(values@a,4.8)
        self.assertAlmostEqual(values@o,1.8)
        self.assertAlmostEqual(values@(o-a),-3.)


if __name__=='__main__': unittest.main()
