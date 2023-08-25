Path Dependent Battery Degradation Dataset 2

Recorded by: Trishna Raj, trishna.raj@eng.ox.ac.uk
Supervisor: David A. Howey, david.howey@eng.ox.ac.uk
___________________________________________________________________________________________________________________________________________________________________________________________________

Place of tests: Battery Intelligence Lab, University of Oxford
Website: http://howey.eng.ox.ac.uk
Test subjects: 12 x NCR18650BD (nickel cobalt aluminium oxide (NCA) positive electrodes and graphite negative electrodes), 3Ah
Battery tester: Maccor 4200 (cycling and reference performance tests) and Biologic MPG205 (EIS)
Environmental chamber: Binder & Maccor thermal chamber, environmental temperature: 24 degC

Related Material : 
	Path Dependent Battery Degradation Dataset 1 :  DOI:10.5287/bodleian:v0ervBv6p
	"Investigation of Path Dependent Degradation in Lithium-Ion Batteries", Batteries & Supercaps (Wiley), Trishna Raj, Andrew A. Wang, Charles W. Monroe and David A. Howey 
___________________________________________________________________________________________________________________________________________________________________________________________________

PLEASE READ THE FOLLOWING MESSAGE CAREFULLY BEFORE PROCEEDING FURTHER:
If you make use of our data, please cite our dataset directly using its DOI.
The data are stored in ".mat" files, which is the Matlab binary file format. You need Matlab, available from Mathworks, to open this type of file.
NOTE: The zip data files are large (approx. 0.8GB)

For raw data files in excel/MIMS client friendly formats with extended testing information, please contact david.howey@eng.ox.ac.uk

Thanks for your interest in our work.
David Howey 
Trishna Raj
Feb 2021
___________________________________________________________________________________________________________________________________________________________________________________________________

A continuation of the 'Path Dependent Battery Degradation Dataset 1' dataset. This dataset contains 4 groups of long term degradation data collected for three 18650 NCA cells per each group after the middle of life (MoL).  Group 5 includes data collected from 3 cells that are exposed to continuous CC cycling at C/2. Group 6 consists of a single cell exposed to continuous calendar aging at 90% SoC. All datafiles include time, current, voltage, capacity and temperature data. All tests were conducted in a thermal chamber set to 24 degC.

EIS data collected at the beginning of life (BoL),MoL and end of life (EoL) for all groups (no MoL EIS data for group 5 cells).

File name format: TPGx.y - Cell z   (x: group number, y:file number , z: cell number)
Description of combined profile groups:
	Group 1 - 1 day of cycling at C/2 and 5 days of calendar aging at 90% SoC
	Group 2 - 1 day of cycling at C/4 and 5 days of calendar aging at 90% SoC 
	Group 3 - 2 days of cycling at C/2 and 10 days of calendar aging at 90% SoC
	Group 4 - 2 days of cycling at C/4 and 10 days of calendar aging at 90% SoC
	Group 5 - CC cycling from 2.5V-4.2V at C/2
	Group 6 - Calendar aging at 90% SoC 

Reference performance tests were conducted every 48 cycles for groups 1-4 and consist of 
	CC-CV discharge at C/2 to 2.5V (step 4)
	CC-CV charge at C/2 to 4.2V until current was C/60 (step 6)
	Pseudo OCV discharge at C/24 to 2.5V (step 8)
	Pseudo OCV charge at C/24 to 4.2V (step 10)
	Pulse characterisation at 80% SoC (steps 14, 16 and 18)
	Pulse characterisation at 50% SoC (steps 22, 24 and 26)
	Pulse characterisation at 20% SoC (steps 30, 32 and 34)

EIS data was collected at BoL and MoL  at 80%, 50% and 20% SoC. EIS was conducted with a 100 mA peak-to-peak excitation amplitude in the frequency range of 5 kHz to 10 mHz, with a resolution of 6 points per decade.
The additional document 'Guide to Datafiles' will help in identifying cycling and reference performance test data for each cell under test.
___________________________________________________________________________________________________________________________________________________________________________________________________

These data are copyright (c) 2021, The Chancellor, Masters and Scholars of the University of Oxford, and the 'Path Dependent Battery Degradation Dataset 2' researchers. All rights reserved.
	
THIS DATA IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS DATA, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
