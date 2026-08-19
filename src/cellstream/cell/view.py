"""Thin AnnData-like view exposing the .X[rows] -> csr contract (state3's backend)."""

from __future__ import annotations


class _XProxy:
    def __init__(self, store):
        self._store = store

    def __getitem__(self, rows):
        return self._store.gather_rows(rows)


class AnnDataView:
    def __init__(self, store):
        self._store = store
        self.X = _XProxy(store)

    @property
    def obs(self):
        return self._store.obs

    @property
    def var(self):
        return self._store.var

    @property
    def uns(self):
        return self._store.uns

    @property
    def obsm(self):
        return self._store.obsm

    @property
    def varm(self):
        return self._store.varm

    @property
    def obsp(self):
        return self._store.obsp

    @property
    def varp(self):
        return self._store.varp

    @property
    def shape(self):
        return self._store.shape

    @property
    def n_obs(self):
        return self._store.n_obs

    @property
    def n_vars(self):
        return self._store.n_vars
