import json
import unittest
import numpy as np
import study


class SweepTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261015)
        self.x = (rng.normal(size=(12, 4, 30)) * np.array([1, 2, .3, .1])[None, :, None] * 1e-5).astype(np.float32)
        self.ids = np.repeat([1, 2, 3], 4)

    def test_archived_batch_arithmetic(self):
        old = study.load_archive()
        for policy in ['absolute', 'relative']:
            for eps in [0, 1e-14, 1e-8, 1e-6]:
                actual, diag = study.batch(self.x, policy, eps)
                expected, prior = old.batch_align(self.x, policy, eps)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(diag['clipped'], prior['clipped'])

    def test_archived_pipeline_arithmetic(self):
        old = study.load_archive()
        for pipeline in ['subject', 'dataset_subject']:
            a, _ = study.align(self.x, self.ids, 'absolute', pipeline, 1e-8, 1e-8)
            b, _ = old.align(self.x, self.ids, 'coherent_absolute', pipeline, 1.)
            np.testing.assert_array_equal(a, b)

    def test_coherent_units(self):
        for pipeline in ['subject', 'dataset_subject']:
            a, _ = study.align(self.x, self.ids, 'absolute', pipeline, 1e-8, 1e-8)
            b, _ = study.align(self.x * np.float32(1e-6), self.ids, 'absolute', pipeline, 1e-20, 1e-8)
            self.assertLess(np.linalg.norm(a.astype(float)-b) / np.linalg.norm(a), 1e-5)

    def test_relative_units(self):
        a, _ = study.align(self.x, self.ids, 'relative', 'dataset_subject', 1e-8, 1e-8)
        b, _ = study.align(self.x*np.float32(1e6), self.ids, 'relative', 'dataset_subject', 1e-8, 1e-8)
        self.assertLess(np.linalg.norm(a.astype(float)-b) / np.linalg.norm(a), 1e-5)

    def test_zero_floor_whitens(self):
        z, d = study.batch(self.x, 'absolute', 0.)
        r = np.mean(np.einsum('nct,ndt->ncd', z.astype(float), z.astype(float)), axis=0)
        np.testing.assert_allclose(r, np.eye(4), atol=1e-6)
        self.assertEqual(d['clipped'], 0)

    def test_exact_rotation(self):
        x = self.x.astype(float)
        r = np.mean(np.einsum('nct,ndt->ncd', x, x), axis=0)
        def invsqrt(a):
            e,v=np.linalg.eigh(a)
            return (v*e**-.5)@v.T
        d = np.diag([1.,2.,3.,4.])
        s = invsqrt(r)
        h = invsqrt(d@r@d)@d
        q = h@np.linalg.inv(s)
        np.testing.assert_allclose(q@q.T,np.eye(4),atol=1e-10)
        np.testing.assert_allclose(h,q@s,atol=1e-8)

    def test_non_spd_rejected(self):
        with self.assertRaises(ValueError): study.batch(np.zeros_like(self.x), 'absolute', 1e-8)

    def test_design_and_anchor(self):
        cfg=json.loads((study.HOME/'config.json').read_text(encoding='utf-8'))
        conditions=study.configurations(cfg)
        self.assertEqual(len(conditions),31)
        self.assertEqual(conditions[0]['policy'],'relative')
        self.assertEqual(conditions[2]['raw_epsilon'],1e-8)
        self.assertFalse(cfg['target_label_parameter_selection'])


if __name__ == '__main__': unittest.main()
