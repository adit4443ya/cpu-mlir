// vl_probe.c: print the CPU's SVE / SME features and vector lengths (run on the phone).
// SSVE only pays when the streaming vector length (SVL) is larger than the SVE one.
#include <stdio.h>
#include <sys/auxv.h>
#define HWCAP_SVE_ (1UL << 22)
#define HWCAP2_SVE2_ (1UL << 1)
#define HWCAP2_SME_ (1UL << 23)
#define HWCAP2_SME_FA64_ (1UL << 30)
int main(void) {
  unsigned long hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
  int sve = !!(hw & HWCAP_SVE_), sve2 = !!(hw2 & HWCAP2_SVE2_), sme = !!(hw2 & HWCAP2_SME_);
  printf("sve=%d sve2=%d sme=%d sme_fa64=%d\n", sve, sve2, sme, !!(hw2 & HWCAP2_SME_FA64_));
  if (sve) { unsigned long vl; __asm__ volatile("rdvl %0, #1" : "=r"(vl)); printf("sve_vl_bits=%lu\n", vl * 8); }
  if (sme) { unsigned long svl; __asm__ volatile("rdsvl %0, #1" : "=r"(svl)); printf("sme_svl_bits=%lu\n", svl * 8); }
  return 0;
}
