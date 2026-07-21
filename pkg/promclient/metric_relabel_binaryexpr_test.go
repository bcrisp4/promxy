package promclient

import (
	"context"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/prometheus/common/model"
	"github.com/prometheus/prometheus/model/relabel"
	"github.com/prometheus/prometheus/promql/parser"
	"github.com/prometheus/prometheus/storage"
	"github.com/prometheus/prometheus/tsdb/chunkenc"
)

// rewriteQuery walks a query with MetricsRelabelVisitor as
// MetricsRelabelClient.Query does, and returns the rewritten query string
// alongside the visitor's badLabel flag. RelabelConfigs is left nil because
// Visit only ever consults MetricsRelabelConfigs; the result-side configs are
// applied by the client, not the visitor.
func rewriteQuery(t *testing.T, cfgs []*MetricRelabelConfig, query string) (string, bool, error) {
	t.Helper()

	e, err := parser.ParseExpr(query)
	if err != nil {
		t.Fatalf("unable to parse %q: %v", query, err)
	}

	v := NewMetricsRelabelVisitor(cfgs, nil)
	if _, err := parser.Walk(context.TODO(), v, &parser.EvalStmt{Expr: e}, e, nil, nil); err != nil {
		return "", false, err
	}

	return e.String(), v.badLabel, nil
}

// TestMetricsRelabelVisitorBinaryExpr covers BinaryExprs where neither side is a
// literal but there is no vector matching to reverse -- e.g. `time() - foo`.
// proxystorage serialises such subtrees into a single downstream query, both via
// the reentrant-aggregation path and via the Call path (the `abs(...)` case
// below needs no aggregation at all), so the visitor has to be able to rewrite
// the selectors inside them.
func TestMetricsRelabelVisitorBinaryExpr(t *testing.T) {
	replace := []*MetricRelabelConfig{
		{
			SourceLabel: model.LabelName("__tenant_id__"),
			TargetLabel: "tenant",
			Action:      relabel.Replace,
		},
	}
	labeldrop := []*MetricRelabelConfig{
		{
			SourceLabel: model.LabelName("__tenant_id__"),
			Action:      relabel.LabelDrop,
		},
	}

	tests := []struct {
		cfgs []*MetricRelabelConfig
		in   string
		out  string
	}{
		// The reported failure: scalar function minus a matrix selector, under a
		// reentrant aggregation.
		{
			cfgs: labeldrop,
			in:   `max(time() - min_over_time(server_start_ts{host=~"h1",service="backend_reader"}[1m]))`,
			out:  `max(time() - min_over_time(server_start_ts{host=~"h1",service="backend_reader"}[1m]))`,
		},
		// Same shape, with a matcher that actually needs rewriting.
		{
			cfgs: replace,
			in:   `max(time() - min_over_time(server_start_ts{tenant="a"}[1m]))`,
			out:  `max(time() - min_over_time(server_start_ts{__tenant_id__="a"}[1m]))`,
		},
		// Reversed operand order.
		{
			cfgs: replace,
			in:   `sum(foo{tenant="a"} - time())`,
			out:  `sum(foo{__tenant_id__="a"} - time())`,
		},
		// scalar() on the right-hand side is still a Call, not a literal.
		{
			cfgs: replace,
			in:   `max(foo{tenant="a"} / scalar(bar{tenant="b"}))`,
			out:  `max(foo{__tenant_id__="a"} / scalar(bar{__tenant_id__="b"}))`,
		},
		// Bare (un-aggregated) form -- reachable via the Call pushdown path.
		{
			cfgs: replace,
			in:   `abs(time() - foo{tenant="a"})`,
			out:  `abs(time() - foo{__tenant_id__="a"})`,
		},
		// Literal-bearing exprs already worked; assert we did not regress them.
		{
			cfgs: replace,
			in:   `max(foo{tenant="a"} > 1)`,
			out:  `max(foo{__tenant_id__="a"} > 1)`,
		},
		// A parenthesised literal is a ParenExpr, not a NumberLiteral, so the
		// old literal check did not see it either. Gating on operand types
		// covers these without a special case.
		{
			cfgs: replace,
			in:   `max(foo{tenant="a"} > (1))`,
			out:  `max(foo{__tenant_id__="a"} > (1))`,
		},
		{
			cfgs: replace,
			in:   `max((foo{tenant="a"}) > (1))`,
			out:  `max((foo{__tenant_id__="a"}) > (1))`,
		},
	}

	for i, test := range tests {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			out, badLabel, err := rewriteQuery(t, test.cfgs, test.in)
			if err != nil {
				t.Fatalf("unexpected error rewriting %q: %v", test.in, err)
			}
			if badLabel {
				t.Fatalf("unexpected badLabel for %q", test.in)
			}
			if out != test.out {
				t.Fatalf("Mismatch in query after rewrite expected=%v actual=%v", test.out, out)
			}
		})
	}
}

// TestMetricsRelabelVisitorBinaryExprAllActions covers every relabel action.
// The BinaryExpr case never reads the relabel configs -- it refuses on the
// shape of the expression alone -- so the trigger is having
// metrics_relabel_configs set at all, not any particular action. All four
// error on the reported query before this change.
func TestMetricsRelabelVisitorBinaryExprAllActions(t *testing.T) {
	const query = `max(time() - min_over_time(server_start_ts[1m]))`

	tests := []*MetricRelabelConfig{
		{SourceLabel: model.LabelName("src"), Action: relabel.LabelDrop},
		{SourceLabel: model.LabelName("src"), TargetLabel: "dst", Action: relabel.Replace},
		{SourceLabel: model.LabelName("src"), TargetLabel: "dst", Action: relabel.Lowercase},
		{SourceLabel: model.LabelName("src"), TargetLabel: "dst", Action: relabel.Uppercase},
	}

	for _, cfg := range tests {
		t.Run(string(cfg.Action), func(t *testing.T) {
			out, badLabel, err := rewriteQuery(t, []*MetricRelabelConfig{cfg}, query)
			if err != nil {
				t.Fatalf("unexpected error rewriting %q with action %s: %v", query, cfg.Action, err)
			}
			if badLabel {
				t.Fatalf("unexpected badLabel for action %s", cfg.Action)
			}
			if out != query {
				t.Fatalf("Mismatch in query after rewrite expected=%v actual=%v", query, out)
			}
		})
	}
}

// TestMetricsRelabelVisitorBinaryExprVectorMatching asserts that BinaryExprs
// which carry vector matching are still refused.
//
// Where the labels are explicit -- on()/ignoring(), group_left()/group_right()
// -- reversing the list is not something RewriteLabels can express (it can
// rename or delete, never append). Where they are implicit, as in `foo + bar`,
// matching is on every label except __name__, which a dropped label silently
// changes: the caller sees labels with __tenant_id__ removed and expects two
// series to match, while the downstream still has the label and refuses to
// match them. So refusing is substantively right in both cases, not merely
// conservative.
//
// This test must keep passing: it is the guard against a fix that is too broad.
func TestMetricsRelabelVisitorBinaryExprVectorMatching(t *testing.T) {
	cfgs := []*MetricRelabelConfig{
		{
			SourceLabel: model.LabelName("__tenant_id__"),
			TargetLabel: "tenant",
			Action:      relabel.Replace,
		},
	}

	queries := []string{
		`max(foo * on(job) group_left(inst) bar)`,
		`max(foo or bar)`,
		`max(foo unless bar)`,
		`max(foo and on(job) bar)`,
		`sum(foo / ignoring(inst) bar)`,
		// No explicit matching clause, but the grammar still assigns a default
		// VectorMatching to every vector-vector operation.
		`sum(foo + bar)`,
		// vector() is a Call, not a VectorSelector, so this subtree holds only
		// one selector and proxystorage's vecFinder bail does not apply: it is
		// pushed down whole and does reach the visitor. `or vector(0)` is a
		// common way to guard against empty results, so this is the shape
		// users are most likely to hit.
		`sum(foo or vector(0))`,
		`max(foo unless vector(0))`,
		// Inside a subquery, where the vecFinder bail also does not apply.
		`max_over_time((foo + bar)[5m:1m])`,
		`max_over_time((foo or bar)[5m:1m])`,
	}

	for i, query := range queries {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			if _, _, err := rewriteQuery(t, cfgs, query); err == nil {
				t.Fatalf("expected an error for vector-matching BinaryExpr %q, got none", query)
			}
		})
	}
}

// TestMetricsRelabelVisitorBinaryExprSuperset pins the property the gate
// relies on: keying on operand types is a strict superset of the literal check
// it replaced, so nothing that used to be traversed now errors. A literal
// operand is never an instant vector, so ExprIsLiteral implies the gate
// admits. Each case is asserted to actually contain a literal-bearing
// BinaryExpr first, so the corpus cannot silently go vacuous.
func TestMetricsRelabelVisitorBinaryExprSuperset(t *testing.T) {
	cfgs := []*MetricRelabelConfig{
		{
			SourceLabel: model.LabelName("__tenant_id__"),
			Action:      relabel.LabelDrop,
		},
	}

	queries := []string{
		`foo > 1`,
		`1 > foo`,
		`foo > bool 1`,
		`foo + 1`,
		`foo * 2`,
		`2 / foo`,
		`foo{host="h1"} > 1`,
		`sum(foo) > 1`,
		`max(foo) * 100`,
		`topk(5, foo) > 1`,
		`rate(foo[5m]) > 0.5`,
		`foo offset 5m > 1`,
		`-foo > 1`,
		`foo == bool 1`,
	}

	for i, query := range queries {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			e, err := parser.ParseExpr(query)
			if err != nil {
				t.Fatalf("unable to parse %q: %v", query, err)
			}

			var literalBearing bool
			var walk func(parser.Node)
			walk = func(n parser.Node) {
				if be, ok := n.(*parser.BinaryExpr); ok {
					if ExprIsLiteral(be.LHS) || ExprIsLiteral(be.RHS) {
						literalBearing = true
					}
				}
				for _, c := range parser.Children(n) {
					walk(c)
				}
			}
			walk(e)
			if !literalBearing {
				t.Fatalf("%q contains no literal-bearing BinaryExpr, so it does not test the superset property", query)
			}

			if _, _, err := rewriteQuery(t, cfgs, query); err != nil {
				t.Fatalf("literal-bearing BinaryExpr %q must still traverse, got: %v", query, err)
			}
		})
	}
}

// TestMetricsRelabelClientBinaryExpr drives a scalar-vector BinaryExpr through
// the client and asserts on what the downstream actually receives: the query is
// forwarded with its matchers rewritten. Before this was supported the visitor
// errored first, so the downstream was never called at all.
func TestMetricsRelabelClientBinaryExpr(t *testing.T) {
	const query = `max(time() - min_over_time(server_start_ts{tenant="a"}[1m]))`
	const want = `max(time() - min_over_time(server_start_ts{__tenant_id__="a"}[1m]))`

	api := &recordingAPI{}
	c, err := NewMetricsRelabelClient(api, []*MetricRelabelConfig{
		{
			SourceLabel: model.LabelName("__tenant_id__"),
			TargetLabel: "tenant",
			Action:      relabel.Replace,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	ss := c.Query(context.TODO(), query, time.Now())
	if err := ss.Err(); err != nil {
		t.Fatalf("unexpected error from Query: %v", err)
	}
	if api.query != want {
		t.Fatalf("Mismatch in query sent downstream expected=%v actual=%v", want, api.query)
	}
}

// e2eSamples drains an instant-query SeriesSet into labels -> value.
func e2eSamples(t *testing.T, ss storage.SeriesSet) map[string]float64 {
	t.Helper()

	out := map[string]float64{}
	for ss.Next() {
		s := ss.At()
		it := s.Iterator(nil)
		for it.Next() == chunkenc.ValFloat {
			_, v := it.At()
			out[s.Labels().String()] = v
		}
		// The loop above also exits on an iterator error, which would
		// otherwise be indistinguishable from a series ending normally.
		if err := it.Err(); err != nil {
			t.Fatalf("error iterating series %v: %v", s.Labels(), err)
		}
	}
	if err := ss.Err(); err != nil {
		t.Fatalf("error draining series set: %v", err)
	}

	return out
}

// TestMetricRelabelBinaryExprE2E runs the reported query shape against a real
// promql engine over HTTP, and asserts on the returned *values* -- not just the
// rewritten query string. The fixture holds two series that differ only in
// __tenant_id__, which is the label being dropped.
//
// At t=600s the samples are 100 (tenant a) and 200 (tenant b), so
// `time() - min_over_time(...)` is 500 and 400 respectively.
func TestMetricRelabelBinaryExprE2E(t *testing.T) {
	client, closeFn, err := CreateTestServer(t, "testdata/binaryexpr_relabel.test")
	if err != nil {
		t.Fatal(err)
	}
	defer closeFn()

	ts := model.Time(600000).Time() // 600s -- time() evaluates to 600

	t.Run("labeldrop", func(t *testing.T) {
		c, err := NewMetricsRelabelClient(client, []*MetricRelabelConfig{
			{SourceLabel: model.LabelName("__tenant_id__"), Action: relabel.LabelDrop},
		})
		if err != nil {
			t.Fatal(err)
		}

		t.Run("value", func(t *testing.T) {
			const query = `max(time() - min_over_time(server_start_ts[1m]))`

			got := e2eSamples(t, c.Query(context.TODO(), query, ts))
			want := map[string]float64{"{}": 500}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Mismatch in query result expected=%v actual=%v", want, got)
			}
		})

		t.Run("grouped", func(t *testing.T) {
			const query = `max by (host) (time() - min_over_time(server_start_ts[1m]))`

			got := e2eSamples(t, c.Query(context.TODO(), query, ts))
			want := map[string]float64{
				`{host="h1"}`: 500,
				`{host="h2"}`: 400,
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Mismatch in query result expected=%v actual=%v", want, got)
			}
		})

		// The relabel wrapper must be value-transparent: dropping a label may
		// change the labels that come back, but never the numbers.
		t.Run("transparent", func(t *testing.T) {
			const query = `max(time() - min_over_time(server_start_ts[1m]))`

			want := e2eSamples(t, client.Query(context.TODO(), query, ts))
			got := e2eSamples(t, c.Query(context.TODO(), query, ts))
			// Both sides come from the same helper, so an empty result on both
			// would compare equal and pass without testing anything.
			if len(want) == 0 {
				t.Fatalf("unwrapped client returned no samples; nothing to compare")
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("relabel client changed values: unwrapped=%v wrapped=%v", want, got)
			}
		})

		// The dropped label must not appear in results.
		t.Run("droplabel", func(t *testing.T) {
			const query = `time() - min_over_time(server_start_ts[1m])`

			ss := c.Query(context.TODO(), query, ts)
			if err := ss.Err(); err != nil {
				t.Fatal(err)
			}
			n := 0
			for ss.Next() {
				n++
				if ss.At().Labels().Has("__tenant_id__") {
					t.Fatalf("found dropped label in response: %v", ss.At().Labels())
				}
			}
			if n != 2 {
				t.Fatalf("Mismatch in series count expected=2 actual=%d", n)
			}
		})
	})

	// A Replace config proves the matcher rewrite *inside* a binary expression
	// reaches the right series in a real downstream: the user queries
	// tenant="a", which must be rewritten to __tenant_id__="a" before it is
	// sent, or it would match nothing.
	t.Run("replace", func(t *testing.T) {
		c, err := NewMetricsRelabelClient(client, []*MetricRelabelConfig{
			{
				SourceLabel: model.LabelName("__tenant_id__"),
				TargetLabel: "tenant",
				Action:      relabel.Replace,
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		const query = `max(time() - min_over_time(server_start_ts{tenant="a"}[1m]))`

		got := e2eSamples(t, c.Query(context.TODO(), query, ts))
		want := map[string]float64{"{}": 500} // only tenant a (100), not b (200 -> 400)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("Mismatch in query result expected=%v actual=%v", want, got)
		}
	})
}

// TestMetricRelabelBinaryExprKnownLimits characterises two pre-existing relabel
// behaviours that allowing these BinaryExpr shapes makes reachable from more
// queries. Neither is introduced here: both reproduce on shapes that were
// always traversable (a bare selector, a standalone scalar()), as the
// subtests assert alongside the binary-expression form.
//
// These pin current behaviour rather than endorse it. If either is ever
// addressed, these tests should fail and be updated deliberately.
func TestMetricRelabelBinaryExprKnownLimits(t *testing.T) {
	client, closeFn, err := CreateTestServer(t, "testdata/binaryexpr_relabel.test")
	if err != nil {
		t.Fatal(err)
	}
	defer closeFn()

	ts := model.Time(600000).Time()

	c, err := NewMetricsRelabelClient(client, []*MetricRelabelConfig{
		{SourceLabel: model.LabelName("__tenant_id__"), Action: relabel.LabelDrop},
	})
	if err != nil {
		t.Fatal(err)
	}

	// labelSets returns the label sets of every returned series, in order and
	// without deduplication, so collisions remain visible.
	labelSets := func(query string) []string {
		ss := c.Query(context.TODO(), query, ts)
		var out []string
		for ss.Next() {
			out = append(out, ss.At().Labels().String())
		}
		if err := ss.Err(); err != nil {
			t.Fatalf("error on %q: %v", query, err)
		}
		return out
	}

	// Dropping a label can collapse two distinct downstream series onto the
	// same label set. MapLabelsSeriesSet does not merge or dedup, so both are
	// returned and the consuming engine sees a duplicate label set.
	t.Run("labeldrop-can-collide", func(t *testing.T) {
		for _, query := range []string{
			`time() - min_over_time(dup_ts[1m])`, // newly allowed shape
			`min_over_time(dup_ts[1m])`,          // always allowed -- same result
		} {
			got := labelSets(query)
			want := []string{`{host="h1"}`, `{host="h1"}`}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("Mismatch in label sets for %q expected=%v actual=%v", query, want, got)
			}
		}
	})

	// A matcher on a dropped label cannot match anything downstream, so the
	// visitor sets badLabel and Query returns an empty set for the *whole*
	// expression. PromQL would instead evaluate scalar(empty) as NaN and
	// return `lhs_ts - NaN` per series.
	t.Run("dropped-label-matcher-empties-whole-query", func(t *testing.T) {
		for _, query := range []string{
			`lhs_ts{host="h1"} - scalar(rhs_ts{__tenant_id__="x"})`, // newly allowed shape
			`scalar(rhs_ts{__tenant_id__="x"})`,                     // always allowed -- same result
		} {
			if got := labelSets(query); len(got) != 0 {
				t.Fatalf("Mismatch in series count for %q expected=0 actual=%d (%v)", query, len(got), got)
			}
		}
	})
}
