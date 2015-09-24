// Copyright ©2012 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nmf_test

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
	"github.com/kortschak/nmf"
)

func posNorm(_, _ int, _ float64) float64 { return math.Abs(rand.NormFloat64()) }

func ExampleFactors() {
	rand.Seed(1)

	V := mat64.NewDense(3, 4, []float64{20, 0, 30, 0, 0, 16, 1, 9, 0, 10, 6, 11})
	fmt.Printf("V =\n%.3f\n\n", mat64.Formatted(V))

	categories := 5

	rows, cols := V.Dims()

	Wo := mat64.NewDense(rows, categories, nil)
	Wo.Apply(posNorm, Wo)

	Ho := mat64.NewDense(categories, cols, nil)
	Ho.Apply(posNorm, Ho)

	conf := nmf.Config{
		Tolerance:   1e-5,
		MaxIter:     100,
		MaxOuterSub: 1000,
		MaxInnerSub: 20,
		Limit:       time.Second,
	}

	W, H, ok := nmf.Factors(V, Wo, Ho, conf)

	var P, D mat64.Dense
	P.Mul(W, H)
	D.Sub(V, &P)

	fmt.Printf("Successfully factorised: %v\n\n", ok)
	fmt.Printf("W =\n%.3f\n\nH =\n%.3f\n\n", mat64.Formatted(W), mat64.Formatted(H))
	fmt.Printf("P =\n%.3f\n\n", mat64.Formatted(&P))
	fmt.Printf("delta = %.3f\n", mat64.Norm(&D, 2))

	// Output:
	// V =
	// ⎡20.000   0.000  30.000   0.000⎤
	// ⎢ 0.000  16.000   1.000   9.000⎥
	// ⎣ 0.000  10.000   6.000  11.000⎦
	//
	// Successfully factorised: true
	//
	// W =
	// ⎡ 0.000   0.000   0.000   6.804  17.063⎤
	// ⎢ 0.000   0.000   7.295   0.000   0.014⎥
	// ⎣ 0.000   1.055   4.560   0.000   1.423⎦
	//
	// H =
	// ⎡1.073  0.700  0.432  1.000⎤
	// ⎢0.000  0.000  2.740  5.096⎥
	// ⎢0.000  2.193  0.134  1.234⎥
	// ⎢2.939  0.000  0.003  0.000⎥
	// ⎣0.000  0.000  1.757  0.000⎦
	//
	// P =
	// ⎡20.000   0.000  30.000   0.000⎤
	// ⎢ 0.000  16.000   1.000   9.000⎥
	// ⎣ 0.000  10.000   6.000  11.000⎦
	//
	// delta = 0.000
}
