// Copyright ©2013 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package nmf is an implementation of non-negative matrix factorisation by alternative
// non-negative least squares using projected gradients.
//
// The algorithm for this method is described in:
//
// Chih-Jen Lin (2007) 'Projected grad Methods for Non-negative Matrix Factorization.'
// Neural Computation 19:2756.
package nmf

import (
	"math"
	"time"

	"github.com/gonum/matrix/mat64"
)

// Config determines the behaviour of a Factors call.
type Config struct {
	// Tolerance is the stopping tolerance for the factorisation.
	Tolerance float64

	// MaxIter is the maximum number of iterations performed by the
	// main factorisation loop.
	MaxIter int

	// Limit is the maximum time spent by the factorisation.
	Limit time.Duration

	// MaxOuterSub and MaxInnerSub are the maximum number of iterations
	// the sub-problem will perform in the outer and inner loops.
	MaxOuterSub, MaxInnerSub int
}

// Factors returns matrices W and H that are non-negative factors of V within the
// specified tolerance and computation limits given initial non-negative solutions Wo
// and Ho.
func Factors(V, Wo, Ho *mat64.Dense, c Config) (W, H *mat64.Dense, ok bool) {
	to := time.Now()

	W = Wo
	H = Ho

	var vT, hT, wT mat64.Dense
	hT.TCopy(H)
	wT.TCopy(W)

	var (
		wr, wc = W.Dims()
		hr, hc = H.Dims()

		tmp mat64.Dense
	)

	var vhT mat64.Dense
	gW := mat64.NewDense(wr, wc, nil)
	tmp.Mul(H, &hT)
	gW.Mul(W, &tmp)
	vhT.Mul(V, &hT)
	gW.Sub(gW, &vhT)

	var wTv mat64.Dense
	gH := mat64.NewDense(hr, hc, nil)
	tmp.Reset()
	tmp.Mul(&wT, W)
	gH.Mul(&tmp, H)
	wTv.Mul(&wT, V)
	gH.Sub(gH, &wTv)

	var gHT, gWHT mat64.Dense
	gHT.TCopy(gH)
	gWHT.Stack(gW, &gHT)

	grad := gWHT.Norm(0)
	tolW := math.Max(0.001, c.Tolerance) * grad
	tolH := tolW

	var (
		_ok  bool
		iter int
	)

	decFiltW := func(r, c int, v float64) float64 {
		// decFiltW is applied to gW, so v = gW.At(r, c).
		if v < 0 || W.At(r, c) > 0 {
			return v
		}
		return 0
	}

	decFiltH := func(r, c int, v float64) float64 {
		// decFiltH is applied to gH, so v = gH.At(r, c).
		if v < 0 || H.At(r, c) > 0 {
			return v
		}
		return 0
	}

	for i := 0; i < c.MaxIter; i++ {
		gW.Apply(decFiltW, gW)
		gH.Apply(decFiltH, gH)

		var proj float64
		for _, v := range gW.RawMatrix().Data {
			proj += v * v
		}
		for _, v := range gH.RawMatrix().Data {
			proj += v * v
		}
		proj = math.Sqrt(proj)
		if proj < c.Tolerance*grad || time.Now().Sub(to) > c.Limit {
			break
		}

		vT.TCopy(V)
		hT.TCopy(H)
		wT.TCopy(W)
		W, gW, iter, ok = nnlsSubproblem(&vT, &hT, &wT, tolW, c.MaxOuterSub, c.MaxInnerSub)
		if iter == 0 {
			tolW *= 0.1
		}

		wT.Reset()
		wT.TCopy(W)
		W = &wT

		var gWT mat64.Dense
		gWT.TCopy(gW)
		*gW = gWT

		H, gH, iter, _ok = nnlsSubproblem(V, W, H, tolH, c.MaxOuterSub, c.MaxInnerSub)
		ok = ok && _ok
		if iter == 0 {
			tolH *= 0.1
		}
	}

	return W, H, ok
}

func posFilt(r, c int, v float64) float64 {
	if v > 0 {
		return v
	}
	return 0
}

func nnlsSubproblem(V, W, Ho *mat64.Dense, tol float64, outer, inner int) (H, G *mat64.Dense, i int, ok bool) {
	H = new(mat64.Dense)
	H.Clone(Ho)

	var wT, WtV, WtW mat64.Dense
	wT.TCopy(W)
	WtV.Mul(&wT, V)
	WtW.Mul(&wT, W)

	alpha, beta := 1., 0.1

	decFilt := func(r, c int, v float64) float64 {
		// decFilt is applied to G, so v = G.At(r, c).
		if v < 0 || H.At(r, c) > 0 {
			return v
		}
		return 0
	}

	G = new(mat64.Dense)
	for i = 0; i < outer; i++ {
		G.Mul(&WtW, H)
		G.Sub(G, &WtV)
		G.Apply(decFilt, G)

		if G.Norm(0) < tol {
			break
		}

		var (
			reduce bool
			Hp     *mat64.Dense
			d, dQ  mat64.Dense
		)
		for j := 0; j < inner; j++ {
			var Hn mat64.Dense
			Hn.Scale(alpha, G)
			Hn.Sub(H, &Hn)
			Hn.Apply(posFilt, &Hn)

			d.Sub(&Hn, H)
			dQ.Mul(&WtW, &d)
			dQ.MulElem(&dQ, &d)
			d.MulElem(G, &d)

			sufficient := 0.99*d.Sum()+0.5*dQ.Sum() < 0

			if j == 0 {
				reduce = !sufficient
				Hp = H
			}
			if reduce {
				if sufficient {
					H = &Hn
					ok = true
					break
				} else {
					alpha *= beta
				}
			} else {
				if !sufficient || Hp.Equals(&Hn) {
					H = Hp
					break
				} else {
					alpha /= beta
					Hp = &Hn
				}
			}
		}
	}

	return H, G, i, ok
}
